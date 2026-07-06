#![recursion_limit = "256"]

//! Apple Metal proof surfaces for selected SymLiquid hot loops.
//!
//! This crate is intentionally narrow. It proves a Rust-owned Metal rollout
//! state-update path against the existing CPU reference before any scheduler or
//! CLI claims full Mac-native hot-loop parity.

#[cfg(target_os = "macos")]
mod macos;

use std::collections::BTreeMap;
use std::time::Instant;

use rand::SeedableRng;
use symliquid_core::benchmarks::{
    evaluate_standalone_model, generate_cgs_hard_suite, prepare_training_examples,
    standalone_features, standalone_output_vocab, training_batch_tensor, write_readout_artifact,
    StandaloneTrainConfig, StandaloneTrainReport,
};
use symliquid_core::error::Result;
use symliquid_core::token_superposition::{
    build_token_superposition_dataset, build_token_superposition_report, evaluate_token_readout,
    make_ar_training_batch, make_bag_training_batch, token_feature_table, TokenSuperpositionConfig,
    TokenSuperpositionRunReport,
};
use symliquid_core::train::LinearReadout;

#[cfg(target_os = "macos")]
pub use macos::{
    linear_readout_logits_metal, metal_available, rollout_sequence_metal,
    train_readout_bag_sgd_metal, train_readout_sgd_metal, update_hv_projection_metal,
    update_state_dynamics_sample_metal,
};

#[cfg(not(target_os = "macos"))]
pub fn metal_available() -> bool {
    false
}

#[cfg(not(target_os = "macos"))]
pub fn rollout_sequence_metal(
    _observations: &symliquid_core::tensor::Tensor,
    _state: &symliquid_cuda::rollout_cuda::RolloutState,
    _params: &symliquid_cuda::rollout_cuda::RolloutParams,
    _config: &symliquid_cuda::rollout_cuda::RolloutConfig,
) -> symliquid_core::error::Result<symliquid_cuda::rollout_cuda::RolloutState> {
    Err(symliquid_core::error::SymError::InvalidArgument(
        "Metal rollout is available only on macOS".to_string(),
    ))
}

#[cfg(not(target_os = "macos"))]
pub fn linear_readout_logits_metal(
    _features: &symliquid_core::tensor::Tensor,
    _readout: &symliquid_core::train::LinearReadout,
) -> symliquid_core::error::Result<symliquid_core::tensor::Tensor> {
    Err(symliquid_core::error::SymError::InvalidArgument(
        "Metal readout is available only on macOS".to_string(),
    ))
}

#[cfg(not(target_os = "macos"))]
pub fn train_readout_sgd_metal(
    _features: &symliquid_core::tensor::Tensor,
    _targets: &[usize],
    _readout: &mut symliquid_core::train::LinearReadout,
    _epochs: usize,
    _lr: f32,
    _samples_per_launch: usize,
) -> symliquid_core::error::Result<symliquid_core::train::TrainingTrace> {
    Err(symliquid_core::error::SymError::InvalidArgument(
        "Metal readout training is available only on macOS".to_string(),
    ))
}

#[cfg(not(target_os = "macos"))]
pub fn train_readout_bag_sgd_metal(
    _features: &symliquid_core::tensor::Tensor,
    _target_bags: &[usize],
    _targets_per_sample: usize,
    _readout: &mut symliquid_core::train::LinearReadout,
    _epochs: usize,
    _lr: f32,
    _samples_per_launch: usize,
) -> symliquid_core::error::Result<symliquid_core::train::TrainingTrace> {
    Err(symliquid_core::error::SymError::InvalidArgument(
        "Metal bag readout training is available only on macOS".to_string(),
    ))
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn update_state_dynamics_sample_metal(
    _obs_features: &[f32],
    _hidden_state: &[f32],
    _reservoir_state: &[f32],
    _hidden_target: &[f32],
    _reservoir_target: &[f32],
    _params: &mut symliquid_cuda::rollout_cuda::RolloutParams,
    _config: &symliquid_cuda::rollout_cuda::RolloutConfig,
    _lr: f32,
) -> symliquid_core::error::Result<()> {
    Err(symliquid_core::error::SymError::InvalidArgument(
        "Metal state dynamics training is available only on macOS".to_string(),
    ))
}

#[cfg(not(target_os = "macos"))]
pub fn update_hv_projection_metal(
    _reservoir_state: &[f32],
    _target_hv: &[f32],
    _params: &mut symliquid_cuda::rollout_cuda::RolloutParams,
    _reservoir_dim: usize,
    _lr: f32,
) -> symliquid_core::error::Result<()> {
    Err(symliquid_core::error::SymError::InvalidArgument(
        "Metal HV projection training is available only on macOS".to_string(),
    ))
}

#[derive(Debug, Clone)]
pub struct RolloutMetalProofConfig {
    pub cases: usize,
    pub batch: usize,
    pub steps: usize,
    pub obs_dim: usize,
    pub hidden_dim: usize,
    pub reservoir_dim: usize,
    pub hv_dim: usize,
    pub output_dim: usize,
    pub dt: f32,
    pub alpha: f32,
    pub memory_decay: f32,
    pub tolerance: f32,
    pub readout_epochs: usize,
    pub readout_lr: f32,
    pub state_epochs: usize,
    pub state_lr: f32,
    pub samples_per_launch: usize,
    pub artifact_path: Option<String>,
    pub train_case_offset: usize,
    pub eval_case_offset: usize,
}

pub type RolloutMetalFeatureProofConfig = RolloutMetalProofConfig;

impl Default for RolloutMetalProofConfig {
    fn default() -> Self {
        Self {
            cases: 6,
            batch: 2,
            steps: 3,
            obs_dim: 3,
            hidden_dim: 4,
            reservoir_dim: 5,
            hv_dim: 7,
            output_dim: 4,
            dt: 0.1,
            alpha: 0.25,
            memory_decay: 0.9,
            tolerance: 1.0e-4,
            readout_epochs: 2,
            readout_lr: 0.03,
            state_epochs: 2,
            state_lr: 0.03,
            samples_per_launch: 2,
            artifact_path: None,
            train_case_offset: 0,
            eval_case_offset: 10_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenSuperpositionMetalProofConfig {
    pub vocab_size: usize,
    pub hv_dim: usize,
    pub train_tokens: usize,
    pub train_samples: usize,
    pub eval_samples: usize,
    pub baseline_epochs: usize,
    pub bag_size: usize,
    pub recovery_ratio: f32,
    pub lr: f32,
    pub samples_per_launch: usize,
    pub tolerance: f32,
}

impl Default for TokenSuperpositionMetalProofConfig {
    fn default() -> Self {
        Self {
            vocab_size: 16,
            hv_dim: 32,
            train_tokens: 192,
            train_samples: 24,
            eval_samples: 24,
            baseline_epochs: 2,
            bag_size: 4,
            recovery_ratio: 0.5,
            lr: 0.03,
            samples_per_launch: 4,
            tolerance: 5.0e-4,
        }
    }
}

pub fn token_superposition_metal_readout_proof_report(
    config: &TokenSuperpositionMetalProofConfig,
) -> serde_json::Value {
    let started = std::time::Instant::now();
    if config.vocab_size == 0
        || config.vocab_size > 256
        || config.hv_dim == 0
        || config.train_samples == 0
        || config.eval_samples == 0
        || config.baseline_epochs == 0
        || config.bag_size == 0
        || !config.recovery_ratio.is_finite()
        || config.recovery_ratio <= 0.0
        || config.recovery_ratio >= 1.0
        || config.samples_per_launch == 0
        || config.tolerance <= 0.0
    {
        return token_superposition_failed_report(started, config, "invalid_bounded_config");
    }
    if !metal_available() {
        return serde_json::json!({
            "ok": false,
            "policy": "project_theseus_macos_metal_token_superposition_readout_proof_v0",
            "state": "RED",
            "reason": "no_default_metal_device",
            "native_path": "token_superposition_bag_readout_sgd_plus_ar_recovery",
            "parity_scope": "bounded_private_synthetic_token_superposition_readout",
            "train_token_superposition_parity_claim_allowed": false,
            "full_cli_parity_claim_allowed": false,
            "external_inference_calls": 0,
            "teacher_used": false,
            "public_training_rows": 0,
            "model_promotion_allowed": false
        });
    }

    let dataset = synthetic_token_superposition_dataset(config);
    let feature_table =
        symliquid_core::token_superposition::token_feature_table(&dataset.vocab, config.hv_dim);
    let initial_readout = token_superposition_readout(config);

    let baseline_features = match symliquid_core::token_superposition::make_ar_training_batch(
        &dataset,
        &feature_table,
        config.train_samples,
        0x5453_5400,
    ) {
        Ok(value) => value,
        Err(error) => {
            return token_superposition_failed_report(
                started,
                config,
                &format!("baseline_batch_failed:{error}"),
            )
        }
    };

    let mut cpu_baseline = initial_readout.clone();
    let cpu_baseline_started = std::time::Instant::now();
    let cpu_baseline_trace = match train_readout_sgd_cpu_reference(
        &baseline_features.0,
        &baseline_features.1,
        &mut cpu_baseline,
        config.baseline_epochs,
        config.lr,
    ) {
        Ok(value) => value,
        Err(error) => {
            return token_superposition_failed_report(
                started,
                config,
                &format!("cpu_baseline_train_failed:{error}"),
            )
        }
    };
    let cpu_baseline_ms = cpu_baseline_started.elapsed().as_secs_f64() * 1000.0;
    let cpu_baseline_eval = match symliquid_core::token_superposition::evaluate_token_readout(
        &cpu_baseline,
        &dataset,
        &feature_table,
        config.eval_samples,
    ) {
        Ok(value) => value,
        Err(error) => {
            return token_superposition_failed_report(
                started,
                config,
                &format!("cpu_baseline_eval_failed:{error}"),
            )
        }
    };

    let mut metal_baseline = initial_readout.clone();
    let metal_baseline_started = std::time::Instant::now();
    let metal_baseline_trace = match train_readout_sgd_metal(
        &baseline_features.0,
        &baseline_features.1,
        &mut metal_baseline,
        config.baseline_epochs,
        config.lr,
        config.samples_per_launch,
    ) {
        Ok(value) => value,
        Err(error) => {
            return token_superposition_failed_report(
                started,
                config,
                &format!("metal_baseline_train_failed:{error}"),
            )
        }
    };
    let metal_baseline_ms = metal_baseline_started.elapsed().as_secs_f64() * 1000.0;
    let metal_baseline_eval = match symliquid_core::token_superposition::evaluate_token_readout(
        &metal_baseline,
        &dataset,
        &feature_table,
        config.eval_samples,
    ) {
        Ok(value) => value,
        Err(error) => {
            return token_superposition_failed_report(
                started,
                config,
                &format!("metal_baseline_eval_failed:{error}"),
            )
        }
    };

    let bag_samples = (config.train_samples / config.bag_size.max(1)).max(1);
    let bag_batch = match symliquid_core::token_superposition::make_bag_training_batch(
        &dataset,
        &feature_table,
        bag_samples,
        config.bag_size,
        0x5453_5401,
    ) {
        Ok(value) => value,
        Err(error) => {
            return token_superposition_failed_report(
                started,
                config,
                &format!("bag_batch_failed:{error}"),
            )
        }
    };
    let bag_epochs = ((config.baseline_epochs as f32) * (1.0 - config.recovery_ratio))
        .round()
        .max(1.0) as usize;
    let recovery_epochs = ((config.baseline_epochs as f32) * config.recovery_ratio)
        .round()
        .max(1.0) as usize;

    let mut cpu_variant = initial_readout.clone();
    let cpu_variant_started = std::time::Instant::now();
    let cpu_bag_trace = match train_readout_bag_sgd_cpu_reference(
        &bag_batch.features,
        &bag_batch.target_bags,
        bag_batch.targets_per_sample,
        &mut cpu_variant,
        bag_epochs,
        config.lr,
    ) {
        Ok(value) => value,
        Err(error) => {
            return token_superposition_failed_report(
                started,
                config,
                &format!("cpu_bag_train_failed:{error}"),
            )
        }
    };
    let cpu_recovery_trace = match train_readout_sgd_cpu_reference(
        &baseline_features.0,
        &baseline_features.1,
        &mut cpu_variant,
        recovery_epochs,
        config.lr,
    ) {
        Ok(value) => value,
        Err(error) => {
            return token_superposition_failed_report(
                started,
                config,
                &format!("cpu_recovery_train_failed:{error}"),
            )
        }
    };
    let cpu_variant_ms = cpu_variant_started.elapsed().as_secs_f64() * 1000.0;
    let cpu_variant_eval = match symliquid_core::token_superposition::evaluate_token_readout(
        &cpu_variant,
        &dataset,
        &feature_table,
        config.eval_samples,
    ) {
        Ok(value) => value,
        Err(error) => {
            return token_superposition_failed_report(
                started,
                config,
                &format!("cpu_variant_eval_failed:{error}"),
            )
        }
    };

    let mut metal_variant = initial_readout;
    let metal_variant_started = std::time::Instant::now();
    let metal_bag_trace = match train_readout_bag_sgd_metal(
        &bag_batch.features,
        &bag_batch.target_bags,
        bag_batch.targets_per_sample,
        &mut metal_variant,
        bag_epochs,
        config.lr,
        config.samples_per_launch,
    ) {
        Ok(value) => value,
        Err(error) => {
            return token_superposition_failed_report(
                started,
                config,
                &format!("metal_bag_train_failed:{error}"),
            )
        }
    };
    let metal_recovery_trace = match train_readout_sgd_metal(
        &baseline_features.0,
        &baseline_features.1,
        &mut metal_variant,
        recovery_epochs,
        config.lr,
        config.samples_per_launch,
    ) {
        Ok(value) => value,
        Err(error) => {
            return token_superposition_failed_report(
                started,
                config,
                &format!("metal_recovery_train_failed:{error}"),
            )
        }
    };
    let metal_variant_ms = metal_variant_started.elapsed().as_secs_f64() * 1000.0;
    let metal_variant_eval = match symliquid_core::token_superposition::evaluate_token_readout(
        &metal_variant,
        &dataset,
        &feature_table,
        config.eval_samples,
    ) {
        Ok(value) => value,
        Err(error) => {
            return token_superposition_failed_report(
                started,
                config,
                &format!("metal_variant_eval_failed:{error}"),
            )
        }
    };

    let cpu_variant_logits = match cpu_variant.logits(&baseline_features.0) {
        Ok(value) => value,
        Err(error) => {
            return token_superposition_failed_report(
                started,
                config,
                &format!("cpu_variant_logits_failed:{error}"),
            )
        }
    };
    let metal_variant_logits =
        match linear_readout_logits_metal(&baseline_features.0, &metal_variant) {
            Ok(value) => value,
            Err(error) => {
                return token_superposition_failed_report(
                    started,
                    config,
                    &format!("metal_variant_logits_failed:{error}"),
                )
            }
        };

    let baseline_weight_max_delta = max_abs_delta(&cpu_baseline.weights, &metal_baseline.weights);
    let baseline_bias_max_delta = max_abs_delta(&cpu_baseline.bias, &metal_baseline.bias);
    let variant_weight_max_delta = max_abs_delta(&cpu_variant.weights, &metal_variant.weights);
    let variant_bias_max_delta = max_abs_delta(&cpu_variant.bias, &metal_variant.bias);
    let variant_logits_max_delta =
        max_abs_delta(&cpu_variant_logits.data, &metal_variant_logits.data);
    let baseline_loss_delta =
        (cpu_baseline_eval.combined_ar_loss - metal_baseline_eval.combined_ar_loss).abs();
    let variant_loss_delta =
        (cpu_variant_eval.combined_ar_loss - metal_variant_eval.combined_ar_loss).abs();
    let variant_prediction_agreement =
        prediction_agreement(&cpu_variant_logits, &metal_variant_logits);
    let passed = baseline_weight_max_delta <= config.tolerance
        && baseline_bias_max_delta <= config.tolerance
        && variant_weight_max_delta <= config.tolerance
        && variant_bias_max_delta <= config.tolerance
        && variant_logits_max_delta <= config.tolerance
        && baseline_loss_delta <= config.tolerance
        && variant_loss_delta <= config.tolerance
        && (variant_prediction_agreement - 1.0).abs() <= f32::EPSILON;
    let baseline_kernel_launches = config.baseline_epochs.saturating_mul(
        config
            .train_samples
            .div_ceil(config.samples_per_launch.max(1)),
    );
    let variant_kernel_launches = bag_epochs
        .saturating_mul(bag_samples.div_ceil(config.samples_per_launch.max(1)))
        + recovery_epochs.saturating_mul(
            config
                .train_samples
                .div_ceil(config.samples_per_launch.max(1)),
        );

    serde_json::json!({
        "ok": passed,
        "policy": "project_theseus_macos_metal_token_superposition_readout_proof_v0",
        "state": if passed { "GREEN" } else { "RED" },
        "native_path": "token_superposition_bag_readout_sgd_plus_ar_recovery",
        "native_bag_trainer": "readout_bag_sgd_samples_kernel",
        "native_ar_trainer": "readout_sgd_samples_kernel",
        "native_readout": "linear_readout_logits_kernel",
        "parity_scope": "bounded_private_synthetic_token_superposition_readout",
        "cuda_equivalent": "readout_cuda::train_readout_bag_sgd_cuda + train_readout_sgd_cuda + linear_readout_logits_cuda",
        "metal_source": "crates/symliquid-metal/kernels/rollout_state_update.metal",
        "rust_owner": "crates/symliquid-metal",
        "train_token_superposition_parity_claim_allowed": false,
        "full_cli_parity_claim_allowed": false,
        "model_promotion_allowed": false,
        "symbolic_fallback": false,
        "config": {
            "vocab_size": config.vocab_size,
            "hv_dim": config.hv_dim,
            "train_tokens": config.train_tokens,
            "train_samples": config.train_samples,
            "eval_samples": config.eval_samples,
            "baseline_epochs": config.baseline_epochs,
            "bag_size": config.bag_size,
            "bag_samples": bag_samples,
            "bag_epochs": bag_epochs,
            "recovery_ratio": config.recovery_ratio,
            "recovery_epochs": recovery_epochs,
            "lr": config.lr,
            "samples_per_launch": config.samples_per_launch,
            "tolerance": config.tolerance
        },
        "dataset": {
            "source": "deterministic_private_synthetic_tokens",
            "vocab_size": dataset.summary.vocab_size,
            "train_tokens": dataset.summary.train_tokens,
            "language_eval_tokens": dataset.summary.language_eval_tokens,
            "code_eval_tokens": dataset.summary.code_eval_tokens,
            "public_training_rows": 0
        },
        "baseline": {
            "cpu_train_loss": cpu_baseline_trace.loss,
            "metal_train_loss": metal_baseline_trace.loss,
            "cpu_combined_loss": cpu_baseline_eval.combined_ar_loss,
            "metal_combined_loss": metal_baseline_eval.combined_ar_loss,
            "loss_delta": baseline_loss_delta,
            "weight_max_abs_delta": baseline_weight_max_delta,
            "bias_max_abs_delta": baseline_bias_max_delta,
            "kernel_launches": baseline_kernel_launches
        },
        "variant": {
            "cpu_bag_loss": cpu_bag_trace.loss,
            "metal_bag_loss": metal_bag_trace.loss,
            "cpu_recovery_loss": cpu_recovery_trace.loss,
            "metal_recovery_loss": metal_recovery_trace.loss,
            "cpu_combined_loss": cpu_variant_eval.combined_ar_loss,
            "metal_combined_loss": metal_variant_eval.combined_ar_loss,
            "loss_delta": variant_loss_delta,
            "weight_max_abs_delta": variant_weight_max_delta,
            "bias_max_abs_delta": variant_bias_max_delta,
            "logits_max_abs_delta": variant_logits_max_delta,
            "prediction_agreement": variant_prediction_agreement,
            "kernel_launches": variant_kernel_launches
        },
        "parity_metrics": {
            "baseline_weight_max_abs_delta": baseline_weight_max_delta,
            "baseline_bias_max_abs_delta": baseline_bias_max_delta,
            "variant_weight_max_abs_delta": variant_weight_max_delta,
            "variant_bias_max_abs_delta": variant_bias_max_delta,
            "variant_logits_max_abs_delta": variant_logits_max_delta,
            "baseline_loss_delta": baseline_loss_delta,
            "variant_loss_delta": variant_loss_delta,
            "prediction_agreement": variant_prediction_agreement
        },
        "timing": {
            "cpu_baseline_wall_ms": cpu_baseline_ms,
            "metal_baseline_wall_ms": metal_baseline_ms,
            "cpu_variant_wall_ms": cpu_variant_ms,
            "metal_variant_wall_ms": metal_variant_ms,
            "report_wall_ms": started.elapsed().as_secs_f64() * 1000.0
        },
        "kernel_launches": baseline_kernel_launches + variant_kernel_launches,
        "guardrails": {
            "no_public_calibration": true,
            "no_public_training_rows": true,
            "no_teacher": true,
            "no_external_inference": true,
            "no_fallback_returns": true,
            "does_not_claim_full_kernel_parity": true,
            "does_not_claim_training_lane_parity": true,
            "does_not_route_scheduler_to_metal": true
        },
        "external_inference_calls": 0,
        "teacher_used": false,
        "public_training_rows": 0
    })
}

pub fn train_token_superposition_metal(
    root: &std::path::Path,
    config: TokenSuperpositionConfig,
    samples_per_launch: usize,
) -> Result<symliquid_core::token_superposition::TokenSuperpositionReport> {
    let total_start = Instant::now();
    let mut timing_breakdown_ms = BTreeMap::new();

    let dataset_start = Instant::now();
    let dataset = build_token_superposition_dataset(root, &config)?;
    let feature_table = token_feature_table(&dataset.vocab, config.hv_dim);
    timing_breakdown_ms.insert(
        "dataset_and_token_features".to_string(),
        dataset_start.elapsed().as_millis(),
    );

    let labels = dataset.vocab.clone();
    let output_dim = labels.len();
    let mut baseline_rng = rand::rngs::StdRng::seed_from_u64(config.train_seed ^ 0x751_BA5E);
    let mut baseline_readout = LinearReadout::new(config.hv_dim, output_dim, &mut baseline_rng);

    let baseline_feature_start = Instant::now();
    let (baseline_features, baseline_targets) = make_ar_training_batch(
        &dataset,
        &feature_table,
        config.train_samples,
        config.train_seed,
    )?;
    let baseline_feature_ms = baseline_feature_start.elapsed().as_millis();

    let baseline_train_start = Instant::now();
    let baseline_trace = train_readout_sgd_metal(
        &baseline_features,
        &baseline_targets,
        &mut baseline_readout,
        config.baseline_epochs,
        config.lr,
        samples_per_launch,
    )?;
    let baseline_train_ms = baseline_train_start.elapsed().as_millis();

    let baseline_eval_start = Instant::now();
    let baseline_eval = evaluate_token_readout(
        &baseline_readout,
        &dataset,
        &feature_table,
        config.eval_samples,
    )?;
    let baseline_eval_ms = baseline_eval_start.elapsed().as_millis();
    let baseline_examples_seen = config.train_samples.saturating_mul(config.baseline_epochs);
    let baseline_kernel_launches = config
        .baseline_epochs
        .saturating_mul(baseline_features.rows.div_ceil(samples_per_launch.max(1)));
    let baseline = TokenSuperpositionRunReport {
        id: "baseline_ar_metal".to_string(),
        objective: "ordinary_ar_next_token".to_string(),
        bag_size: None,
        recovery_ratio: None,
        bag_epochs: 0,
        recovery_epochs: config.baseline_epochs,
        baseline_epochs: config.baseline_epochs,
        train_samples: config.train_samples,
        bag_samples: 0,
        recovery_samples: config.train_samples,
        train_runtime_ms: baseline_train_ms,
        feature_build_ms: baseline_feature_ms,
        eval_runtime_ms: baseline_eval_ms,
        total_runtime_ms: baseline_feature_ms + baseline_train_ms + baseline_eval_ms,
        train_examples_seen: baseline_examples_seen,
        train_examples_per_second: examples_per_second(baseline_examples_seen, baseline_train_ms),
        kernel_launches: baseline_kernel_launches,
        train_loss: baseline_trace.loss,
        train_accuracy: baseline_trace.accuracy,
        eval: baseline_eval,
        nominal_speedup_vs_baseline: 1.0,
        measured_train_speedup_vs_baseline: 1.0,
        measured_total_speedup_vs_baseline: 1.0,
        combined_loss_delta_vs_baseline: 0.0,
        code_loss_delta_vs_baseline: 0.0,
    };

    let mut variants_with_readouts = Vec::new();
    for bag_size in config.bag_sizes.iter().copied().filter(|size| *size > 0) {
        for recovery_ratio in config
            .recovery_ratios
            .iter()
            .copied()
            .filter(|ratio| ratio.is_finite() && *ratio > 0.0 && *ratio < 1.0)
        {
            let (variant, readout) = train_token_superposition_variant_metal(
                &config,
                &dataset,
                &feature_table,
                output_dim,
                samples_per_launch,
                &baseline,
                &baseline_features,
                &baseline_targets,
                bag_size,
                recovery_ratio,
            )?;
            variants_with_readouts.push((variant, readout));
        }
    }
    let artifact_readout = variants_with_readouts
        .iter()
        .min_by(|(left, _), (right, _)| {
            left.eval
                .combined_ar_loss
                .total_cmp(&right.eval.combined_ar_loss)
                .then_with(|| {
                    right
                        .nominal_speedup_vs_baseline
                        .total_cmp(&left.nominal_speedup_vs_baseline)
                })
        })
        .map(|(_, readout)| readout)
        .unwrap_or(&baseline_readout);
    if let Some(path) = config
        .artifact_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        write_readout_artifact(
            path,
            artifact_readout,
            &labels,
            config.hv_dim,
            "metal_token_superposition_readout_private_residual_train_eval",
        )?;
    }
    let variants = variants_with_readouts
        .into_iter()
        .map(|(report, _)| report)
        .collect::<Vec<_>>();

    timing_breakdown_ms.insert("total".to_string(), total_start.elapsed().as_millis());
    Ok(build_token_superposition_report(
        "apple_metal",
        false,
        config,
        dataset.summary,
        baseline,
        variants,
        timing_breakdown_ms,
    ))
}

#[allow(clippy::too_many_arguments)]
fn train_token_superposition_variant_metal(
    config: &TokenSuperpositionConfig,
    dataset: &symliquid_core::token_superposition::TokenSuperpositionDataset,
    feature_table: &[Vec<f32>],
    output_dim: usize,
    samples_per_launch: usize,
    baseline: &TokenSuperpositionRunReport,
    recovery_features: &symliquid_core::tensor::Tensor,
    recovery_targets: &[usize],
    bag_size: usize,
    recovery_ratio: f32,
) -> Result<(TokenSuperpositionRunReport, LinearReadout)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.train_seed ^ 0x7A57_0001);
    let mut readout = LinearReadout::new(config.hv_dim, output_dim, &mut rng);
    let bag_epochs = ((config.baseline_epochs as f32) * (1.0 - recovery_ratio))
        .round()
        .max(1.0) as usize;
    let recovery_epochs = ((config.baseline_epochs as f32) * recovery_ratio)
        .round()
        .max(1.0) as usize;
    let bag_samples = (config.train_samples / bag_size.max(1)).max(1);

    let feature_start = Instant::now();
    let bag_batch = make_bag_training_batch(
        dataset,
        feature_table,
        bag_samples,
        bag_size,
        config.train_seed ^ ((bag_size as u64) << 16) ^ ((recovery_ratio * 1000.0) as u64),
    )?;
    let feature_build_ms = feature_start.elapsed().as_millis();

    let train_start = Instant::now();
    let bag_trace = train_readout_bag_sgd_metal(
        &bag_batch.features,
        &bag_batch.target_bags,
        bag_batch.targets_per_sample,
        &mut readout,
        bag_epochs,
        config.lr,
        samples_per_launch,
    )?;
    let recovery_trace = train_readout_sgd_metal(
        recovery_features,
        recovery_targets,
        &mut readout,
        recovery_epochs,
        config.lr,
        samples_per_launch,
    )?;
    let train_runtime_ms = train_start.elapsed().as_millis();

    let eval_start = Instant::now();
    let eval = evaluate_token_readout(&readout, dataset, feature_table, config.eval_samples)?;
    let eval_runtime_ms = eval_start.elapsed().as_millis();
    let train_examples_seen = bag_samples.saturating_mul(bag_epochs)
        + config.train_samples.saturating_mul(recovery_epochs);
    let baseline_examples_seen = baseline.train_examples_seen.max(1);
    let kernel_launches = bag_epochs
        .saturating_mul(bag_batch.features.rows.div_ceil(samples_per_launch.max(1)))
        + recovery_epochs
            .saturating_mul(recovery_features.rows.div_ceil(samples_per_launch.max(1)));
    let report = TokenSuperpositionRunReport {
        id: format!("tst_s{bag_size}_r{recovery_ratio:.2}_metal"),
        objective: "token_superposition_bag_ce_then_ar_recovery".to_string(),
        bag_size: Some(bag_size),
        recovery_ratio: Some(recovery_ratio),
        bag_epochs,
        recovery_epochs,
        baseline_epochs: config.baseline_epochs,
        train_samples: config.train_samples,
        bag_samples,
        recovery_samples: config.train_samples,
        train_runtime_ms,
        feature_build_ms,
        eval_runtime_ms,
        total_runtime_ms: feature_build_ms + train_runtime_ms + eval_runtime_ms,
        train_examples_seen,
        train_examples_per_second: examples_per_second(train_examples_seen, train_runtime_ms),
        kernel_launches,
        train_loss: 0.5 * (bag_trace.loss + recovery_trace.loss),
        train_accuracy: 0.5 * (bag_trace.accuracy + recovery_trace.accuracy),
        eval: eval.clone(),
        nominal_speedup_vs_baseline: baseline_examples_seen as f32
            / train_examples_seen.max(1) as f32,
        measured_train_speedup_vs_baseline: baseline.train_runtime_ms.max(1) as f32
            / train_runtime_ms.max(1) as f32,
        measured_total_speedup_vs_baseline: baseline.total_runtime_ms.max(1) as f32
            / (feature_build_ms + train_runtime_ms + eval_runtime_ms).max(1) as f32,
        combined_loss_delta_vs_baseline: baseline.eval.combined_ar_loss - eval.combined_ar_loss,
        code_loss_delta_vs_baseline: baseline.eval.code_ar_loss - eval.code_ar_loss,
    };
    Ok((report, readout))
}

pub fn rollout_metal_feature_proof_report(
    config: &RolloutMetalFeatureProofConfig,
) -> serde_json::Value {
    let started = std::time::Instant::now();
    if config.batch == 0
        || config.steps == 0
        || config.obs_dim == 0
        || config.hidden_dim == 0
        || config.reservoir_dim == 0
        || config.hv_dim == 0
    {
        return feature_failed_report(started, config, "feature proof dimensions must be nonzero");
    }
    if !metal_available() {
        return serde_json::json!({
            "ok": false,
            "policy": "project_theseus_macos_metal_rollout_feature_path_proof_v0",
            "state": "RED",
            "reason": "no_default_metal_device",
            "native_path": "rollout_memory_feature_tensor",
            "parity_claim_allowed": false,
            "full_cli_parity_claim_allowed": false,
            "train_rollout_parity_claim_allowed": false,
            "external_inference_calls": 0,
            "teacher_used": false,
            "public_training_rows": 0
        });
    }

    let params = rollout_params(config);
    let cpu_started = std::time::Instant::now();
    let cpu = match rollout_feature_tensor(config, &params, rollout_backend_cpu) {
        Ok(value) => value,
        Err(error) => {
            return feature_failed_report(
                started,
                config,
                &format!("cpu_feature_path_failed:{error}"),
            )
        }
    };
    let cpu_wall_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;

    let metal_started = std::time::Instant::now();
    let metal = match rollout_feature_tensor(config, &params, rollout_backend_metal) {
        Ok(value) => value,
        Err(error) => {
            return feature_failed_report(
                started,
                config,
                &format!("metal_feature_path_failed:{error}"),
            )
        }
    };
    let metal_wall_ms = metal_started.elapsed().as_secs_f64() * 1000.0;

    let max_delta = max_abs_delta(&cpu.data, &metal.data);
    let row_max_delta = max_row_abs_delta(&cpu, &metal);
    let passed = max_delta <= config.tolerance;
    let chunks = config.cases.div_ceil(config.batch);

    serde_json::json!({
        "ok": passed,
        "policy": "project_theseus_macos_metal_rollout_feature_path_proof_v0",
        "state": if passed { "GREEN" } else { "RED" },
        "native_path": "rollout_memory_feature_tensor",
        "native_hot_loop": "rollout_state_update",
        "parity_scope": "bounded_private_synthetic_rollout_feature_batch",
        "parity_claim_allowed": false,
        "full_cli_parity_claim_allowed": false,
        "train_rollout_parity_claim_allowed": false,
        "cuda_equivalent": "crates/symliquid-cuda/src/rollout_cuda.rs::rollout_features_cuda over rollout_state_update_kernel",
        "metal_source": "crates/symliquid-metal/kernels/rollout_state_update.metal",
        "rust_owner": "crates/symliquid-metal",
        "tolerance": config.tolerance,
        "max_abs_delta": max_delta,
        "row_max_abs_delta": row_max_delta,
        "feature_tensor": {
            "rows": config.cases,
            "cols": config.hv_dim,
            "values": config.cases.saturating_mul(config.hv_dim),
            "source": "deterministic_private_synthetic_sequences"
        },
        "config": {
            "cases": config.cases,
            "batch": config.batch,
            "obs_dim": config.obs_dim,
            "hidden_dim": config.hidden_dim,
            "reservoir_dim": config.reservoir_dim,
            "hv_dim": config.hv_dim,
            "steps": config.steps,
            "dt": config.dt,
            "alpha": config.alpha,
            "memory_decay": config.memory_decay
        },
        "timing": {
            "cpu_wall_ms": cpu_wall_ms,
            "metal_wall_ms": metal_wall_ms,
            "report_wall_ms": started.elapsed().as_secs_f64() * 1000.0
        },
        "kernel_launches": chunks.saturating_mul(config.steps),
        "guardrails": {
            "no_public_calibration": true,
            "no_public_training_rows": true,
            "no_teacher": true,
            "no_external_inference": true,
            "does_not_claim_full_kernel_parity": true,
            "does_not_claim_training_parity": true,
            "does_not_route_scheduler_to_metal": true
        },
        "external_inference_calls": 0,
        "teacher_used": false,
        "public_training_rows": 0,
        "model_promotion_allowed": false
    })
}

pub fn rollout_metal_readout_proof_report(config: &RolloutMetalProofConfig) -> serde_json::Value {
    let started = std::time::Instant::now();
    if config.output_dim == 0 {
        return readout_failed_report(started, config, "readout proof output_dim must be nonzero");
    }
    if !metal_available() {
        return serde_json::json!({
            "ok": false,
            "policy": "project_theseus_macos_metal_rollout_readout_proof_v0",
            "state": "RED",
            "reason": "no_default_metal_device",
            "native_path": "rollout_memory_feature_tensor_plus_linear_readout_logits",
            "parity_claim_allowed": false,
            "full_cli_parity_claim_allowed": false,
            "train_rollout_parity_claim_allowed": false,
            "external_inference_calls": 0,
            "teacher_used": false,
            "public_training_rows": 0
        });
    }

    let params = rollout_params(config);
    let cpu_features = match rollout_feature_tensor(config, &params, rollout_backend_cpu) {
        Ok(value) => value,
        Err(error) => {
            return readout_failed_report(
                started,
                config,
                &format!("cpu_feature_path_failed:{error}"),
            )
        }
    };
    let metal_features = match rollout_feature_tensor(config, &params, rollout_backend_metal) {
        Ok(value) => value,
        Err(error) => {
            return readout_failed_report(
                started,
                config,
                &format!("metal_feature_path_failed:{error}"),
            )
        }
    };
    let readout = deterministic_readout(config);

    let cpu_started = std::time::Instant::now();
    let cpu_logits = match readout.logits(&cpu_features) {
        Ok(value) => value,
        Err(error) => {
            return readout_failed_report(started, config, &format!("cpu_logits_failed:{error}"))
        }
    };
    let cpu_wall_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;

    let metal_started = std::time::Instant::now();
    let metal_logits = match linear_readout_logits_metal(&metal_features, &readout) {
        Ok(value) => value,
        Err(error) => {
            return readout_failed_report(started, config, &format!("metal_logits_failed:{error}"))
        }
    };
    let metal_wall_ms = metal_started.elapsed().as_secs_f64() * 1000.0;

    let feature_max_delta = max_abs_delta(&cpu_features.data, &metal_features.data);
    let logits_max_delta = max_abs_delta(&cpu_logits.data, &metal_logits.data);
    let prediction_agreement = prediction_agreement(&cpu_logits, &metal_logits);
    let targets = deterministic_targets(config.cases, config.output_dim);
    let cpu_accuracy = accuracy(&cpu_logits, &targets);
    let metal_accuracy = accuracy(&metal_logits, &targets);
    let passed = feature_max_delta <= config.tolerance
        && logits_max_delta <= config.tolerance
        && (prediction_agreement - 1.0).abs() <= f32::EPSILON;

    serde_json::json!({
        "ok": passed,
        "policy": "project_theseus_macos_metal_rollout_readout_proof_v0",
        "state": if passed { "GREEN" } else { "RED" },
        "native_path": "rollout_memory_feature_tensor_plus_linear_readout_logits",
        "native_hot_loop": "rollout_state_update",
        "native_readout": "linear_readout_logits_kernel",
        "parity_scope": "bounded_private_synthetic_rollout_feature_batch_plus_readout_logits",
        "parity_claim_allowed": false,
        "full_cli_parity_claim_allowed": false,
        "train_rollout_parity_claim_allowed": false,
        "cuda_equivalent": "rollout_features_cuda + readout_cuda::linear_readout_logits_cuda",
        "metal_source": "crates/symliquid-metal/kernels/rollout_state_update.metal",
        "rust_owner": "crates/symliquid-metal",
        "tolerance": config.tolerance,
        "feature_max_abs_delta": feature_max_delta,
        "logits_max_abs_delta": logits_max_delta,
        "prediction_agreement": prediction_agreement,
        "cpu_accuracy_against_deterministic_private_targets": cpu_accuracy,
        "metal_accuracy_against_deterministic_private_targets": metal_accuracy,
        "feature_tensor": {
            "rows": config.cases,
            "cols": config.hv_dim,
            "values": config.cases.saturating_mul(config.hv_dim),
            "source": "deterministic_private_synthetic_sequences"
        },
        "readout": {
            "input_dim": config.hv_dim,
            "output_dim": config.output_dim,
            "parameter_count": readout.parameter_count(),
            "source": "deterministic_private_synthetic_weights"
        },
        "config": {
            "cases": config.cases,
            "batch": config.batch,
            "obs_dim": config.obs_dim,
            "hidden_dim": config.hidden_dim,
            "reservoir_dim": config.reservoir_dim,
            "hv_dim": config.hv_dim,
            "output_dim": config.output_dim,
            "steps": config.steps,
            "dt": config.dt,
            "alpha": config.alpha,
            "memory_decay": config.memory_decay
        },
        "timing": {
            "cpu_logits_wall_ms": cpu_wall_ms,
            "metal_logits_wall_ms": metal_wall_ms,
            "report_wall_ms": started.elapsed().as_secs_f64() * 1000.0
        },
        "guardrails": {
            "no_public_calibration": true,
            "no_public_training_rows": true,
            "no_teacher": true,
            "no_external_inference": true,
            "does_not_claim_full_kernel_parity": true,
            "does_not_claim_training_parity": true,
            "does_not_route_scheduler_to_metal": true
        },
        "external_inference_calls": 0,
        "teacher_used": false,
        "public_training_rows": 0,
        "model_promotion_allowed": false
    })
}

pub fn rollout_metal_readout_training_proof_report(
    config: &RolloutMetalProofConfig,
) -> serde_json::Value {
    let started = std::time::Instant::now();
    if config.output_dim == 0
        || config.readout_epochs == 0
        || config.readout_lr <= 0.0
        || config.samples_per_launch == 0
    {
        return readout_training_failed_report(
            started,
            config,
            "readout training proof requires output_dim, epochs, lr, and samples_per_launch",
        );
    }
    if !metal_available() {
        return serde_json::json!({
            "ok": false,
            "policy": "project_theseus_macos_metal_rollout_readout_training_proof_v0",
            "state": "RED",
            "reason": "no_default_metal_device",
            "native_path": "rollout_memory_feature_tensor_plus_readout_sgd_training_plus_logits_eval",
            "parity_claim_allowed": false,
            "full_cli_parity_claim_allowed": false,
            "train_rollout_parity_claim_allowed": false,
            "external_inference_calls": 0,
            "teacher_used": false,
            "public_training_rows": 0
        });
    }

    let params = rollout_params(config);
    let cpu_features = match rollout_feature_tensor(config, &params, rollout_backend_cpu) {
        Ok(value) => value,
        Err(error) => {
            return readout_training_failed_report(
                started,
                config,
                &format!("cpu_feature_path_failed:{error}"),
            )
        }
    };
    let metal_features = match rollout_feature_tensor(config, &params, rollout_backend_metal) {
        Ok(value) => value,
        Err(error) => {
            return readout_training_failed_report(
                started,
                config,
                &format!("metal_feature_path_failed:{error}"),
            )
        }
    };
    let targets = deterministic_targets(config.cases, config.output_dim);
    let mut cpu_readout = deterministic_readout(config);
    let mut metal_readout = cpu_readout.clone();

    let cpu_started = std::time::Instant::now();
    let cpu_trace = match train_readout_sgd_cpu_reference(
        &cpu_features,
        &targets,
        &mut cpu_readout,
        config.readout_epochs,
        config.readout_lr,
    ) {
        Ok(value) => value,
        Err(error) => {
            return readout_training_failed_report(
                started,
                config,
                &format!("cpu_readout_training_failed:{error}"),
            )
        }
    };
    let cpu_train_wall_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;

    let metal_started = std::time::Instant::now();
    let metal_train_trace = match train_readout_sgd_metal(
        &metal_features,
        &targets,
        &mut metal_readout,
        config.readout_epochs,
        config.readout_lr,
        config.samples_per_launch,
    ) {
        Ok(value) => value,
        Err(error) => {
            return readout_training_failed_report(
                started,
                config,
                &format!("metal_readout_training_failed:{error}"),
            )
        }
    };
    let metal_train_wall_ms = metal_started.elapsed().as_secs_f64() * 1000.0;

    let cpu_logits = match cpu_readout.logits(&cpu_features) {
        Ok(value) => value,
        Err(error) => {
            return readout_training_failed_report(
                started,
                config,
                &format!("cpu_logits_after_training_failed:{error}"),
            )
        }
    };
    let metal_logits = match linear_readout_logits_metal(&metal_features, &metal_readout) {
        Ok(value) => value,
        Err(error) => {
            return readout_training_failed_report(
                started,
                config,
                &format!("metal_logits_after_training_failed:{error}"),
            )
        }
    };
    let metal_eval_trace = match eval_trace_from_logits(&metal_logits, &targets) {
        Ok(value) => value,
        Err(error) => {
            return readout_training_failed_report(
                started,
                config,
                &format!("metal_eval_trace_failed:{error}"),
            )
        }
    };

    let feature_max_delta = max_abs_delta(&cpu_features.data, &metal_features.data);
    let weight_max_delta = max_abs_delta(&cpu_readout.weights, &metal_readout.weights);
    let bias_max_delta = max_abs_delta(&cpu_readout.bias, &metal_readout.bias);
    let logits_max_delta = max_abs_delta(&cpu_logits.data, &metal_logits.data);
    let prediction_agreement = prediction_agreement(&cpu_logits, &metal_logits);
    let eval_loss_delta = (cpu_trace.loss - metal_eval_trace.loss).abs();
    let eval_accuracy_delta = (cpu_trace.accuracy - metal_eval_trace.accuracy).abs();
    let passed = feature_max_delta <= config.tolerance
        && weight_max_delta <= config.tolerance
        && bias_max_delta <= config.tolerance
        && logits_max_delta <= config.tolerance
        && eval_loss_delta <= config.tolerance
        && eval_accuracy_delta <= f32::EPSILON
        && (prediction_agreement - 1.0).abs() <= f32::EPSILON;

    serde_json::json!({
        "ok": passed,
        "policy": "project_theseus_macos_metal_rollout_readout_training_proof_v0",
        "state": if passed { "GREEN" } else { "RED" },
        "native_path": "rollout_memory_feature_tensor_plus_readout_sgd_training_plus_logits_eval",
        "native_hot_loop": "rollout_state_update",
        "native_trainer": "readout_sgd_samples_kernel",
        "native_readout": "linear_readout_logits_kernel",
        "parity_scope": "bounded_private_synthetic_rollout_feature_batch_plus_readout_sgd_training_and_eval",
        "parity_claim_allowed": false,
        "full_cli_parity_claim_allowed": false,
        "train_rollout_parity_claim_allowed": false,
        "readout_training_subpath_proof": true,
        "cuda_equivalent": "rollout_features_cuda + readout_cuda::readout_sgd_samples_kernel + linear_readout_logits_cuda",
        "metal_source": "crates/symliquid-metal/kernels/rollout_state_update.metal",
        "rust_owner": "crates/symliquid-metal",
        "tolerance": config.tolerance,
        "feature_max_abs_delta": feature_max_delta,
        "weight_max_abs_delta": weight_max_delta,
        "bias_max_abs_delta": bias_max_delta,
        "logits_max_abs_delta": logits_max_delta,
        "prediction_agreement": prediction_agreement,
        "eval_loss_delta": eval_loss_delta,
        "eval_accuracy_delta": eval_accuracy_delta,
        "cpu_eval": {
            "loss": cpu_trace.loss,
            "accuracy": cpu_trace.accuracy,
            "grad_norm": cpu_trace.grad_norm
        },
        "metal_eval": {
            "loss": metal_eval_trace.loss,
            "accuracy": metal_eval_trace.accuracy,
            "grad_norm": metal_eval_trace.grad_norm
        },
        "metal_train_trace": {
            "loss": metal_train_trace.loss,
            "accuracy": metal_train_trace.accuracy,
            "grad_norm": metal_train_trace.grad_norm
        },
        "feature_tensor": {
            "rows": config.cases,
            "cols": config.hv_dim,
            "values": config.cases.saturating_mul(config.hv_dim),
            "source": "deterministic_private_synthetic_sequences"
        },
        "targets": {
            "rows": targets.len(),
            "output_dim": config.output_dim,
            "source": "deterministic_private_synthetic_targets"
        },
        "readout": {
            "input_dim": config.hv_dim,
            "output_dim": config.output_dim,
            "parameter_count": cpu_readout.parameter_count(),
            "initialization": "deterministic_private_synthetic_weights"
        },
        "config": {
            "cases": config.cases,
            "batch": config.batch,
            "obs_dim": config.obs_dim,
            "hidden_dim": config.hidden_dim,
            "reservoir_dim": config.reservoir_dim,
            "hv_dim": config.hv_dim,
            "output_dim": config.output_dim,
            "steps": config.steps,
            "dt": config.dt,
            "alpha": config.alpha,
            "memory_decay": config.memory_decay,
            "readout_epochs": config.readout_epochs,
            "readout_lr": config.readout_lr,
            "samples_per_launch": config.samples_per_launch
        },
        "timing": {
            "cpu_train_wall_ms": cpu_train_wall_ms,
            "metal_train_wall_ms": metal_train_wall_ms,
            "report_wall_ms": started.elapsed().as_secs_f64() * 1000.0
        },
        "guardrails": {
            "no_public_calibration": true,
            "no_public_training_rows": true,
            "no_teacher": true,
            "no_external_inference": true,
            "does_not_claim_full_kernel_parity": true,
            "does_not_claim_full_training_parity": true,
            "does_not_route_scheduler_to_metal": true
        },
        "external_inference_calls": 0,
        "teacher_used": false,
        "public_training_rows": 0,
        "model_promotion_allowed": false
    })
}

#[derive(Clone, Copy)]
enum StateTrainingUpdateBackend {
    CpuReference,
    MetalKernels,
}

struct StateTrainingProofTrace {
    wall_ms: f64,
    rollout_kernel_launches: usize,
    state_update_kernel_launches: usize,
    hv_projection_kernel_launches: usize,
}

pub fn rollout_metal_state_training_proof_report(
    config: &RolloutMetalProofConfig,
) -> serde_json::Value {
    let started = std::time::Instant::now();
    if config.cases == 0
        || config.batch == 0
        || config.steps == 0
        || config.output_dim == 0
        || config.state_epochs == 0
        || config.state_lr <= 0.0
    {
        return state_training_failed_report(
            started,
            config,
            "state training proof requires cases, batch, steps, output_dim, state_epochs, and state_lr",
        );
    }
    if !metal_available() {
        return serde_json::json!({
            "ok": false,
            "policy": "project_theseus_macos_metal_rollout_state_training_proof_v0",
            "state": "RED",
            "reason": "no_default_metal_device",
            "native_path": "rollout_state_update_plus_state_dynamics_training_update",
            "parity_scope": "bounded_private_synthetic_rollout_state_training_update",
            "state_training_native_ported": false,
            "cuda_state_training_parity_claim_allowed": false,
            "train_rollout_sweep_parity_claim_allowed": false,
            "full_cli_parity_claim_allowed": false,
            "external_inference_calls": 0,
            "teacher_used": false,
            "public_training_rows": 0,
            "model_promotion_allowed": false
        });
    }

    let base_params = rollout_params(config);
    let mut cpu_candidate = base_params.clone();
    let mut metal_candidate = base_params.clone();
    let cpu_train = match train_state_params_for_proof(
        config,
        &mut cpu_candidate,
        rollout_backend_cpu,
        StateTrainingUpdateBackend::CpuReference,
    ) {
        Ok(value) => value,
        Err(error) => {
            return state_training_failed_report(
                started,
                config,
                &format!("cpu_state_training_failed:{error}"),
            )
        }
    };
    let metal_train = match train_state_params_for_proof(
        config,
        &mut metal_candidate,
        rollout_backend_metal,
        StateTrainingUpdateBackend::MetalKernels,
    ) {
        Ok(value) => value,
        Err(error) => {
            return state_training_failed_report(
                started,
                config,
                &format!("metal_state_training_failed:{error}"),
            )
        }
    };

    let effective_tolerance = config.tolerance.max(5.0e-4);
    let param_deltas = rollout_param_deltas(&cpu_candidate, &metal_candidate);
    let param_max_delta = param_deltas.values().copied().fold(0.0_f32, f32::max);
    let update_delta_from_base = rollout_param_max_delta(&base_params, &metal_candidate);
    let base_eval_features = match rollout_feature_tensor_with_offset(
        config,
        &base_params,
        rollout_backend_cpu,
        config.eval_case_offset,
    ) {
        Ok(value) => value,
        Err(error) => {
            return state_training_failed_report(
                started,
                config,
                &format!("base_eval_feature_failed:{error}"),
            )
        }
    };
    let cpu_eval_features = match rollout_feature_tensor_with_offset(
        config,
        &cpu_candidate,
        rollout_backend_cpu,
        config.eval_case_offset,
    ) {
        Ok(value) => value,
        Err(error) => {
            return state_training_failed_report(
                started,
                config,
                &format!("cpu_candidate_eval_feature_failed:{error}"),
            )
        }
    };
    let metal_eval_features = match rollout_feature_tensor_with_offset(
        config,
        &metal_candidate,
        rollout_backend_metal,
        config.eval_case_offset,
    ) {
        Ok(value) => value,
        Err(error) => {
            return state_training_failed_report(
                started,
                config,
                &format!("metal_candidate_eval_feature_failed:{error}"),
            )
        }
    };
    let targets =
        deterministic_targets_with_offset(config.cases, config.output_dim, config.eval_case_offset);
    let hv_targets = state_label_vectors(config.output_dim, config.hv_dim, 0x4856);
    let base_loss = mean_target_mse(&base_eval_features, &targets, &hv_targets);
    let cpu_candidate_loss = mean_target_mse(&cpu_eval_features, &targets, &hv_targets);
    let metal_candidate_loss = mean_target_mse(&metal_eval_features, &targets, &hv_targets);
    let base_alignment = mean_target_alignment(&base_eval_features, &targets, &hv_targets);
    let cpu_candidate_alignment = mean_target_alignment(&cpu_eval_features, &targets, &hv_targets);
    let metal_candidate_alignment =
        mean_target_alignment(&metal_eval_features, &targets, &hv_targets);
    let cpu_accepts = state_training_candidate_accepts(base_loss, cpu_candidate_loss);
    let metal_accepts = state_training_candidate_accepts(base_loss, metal_candidate_loss);
    let feature_max_delta = max_abs_delta(&cpu_eval_features.data, &metal_eval_features.data);
    let loss_delta = (cpu_candidate_loss - metal_candidate_loss).abs();
    let alignment_delta = (cpu_candidate_alignment - metal_candidate_alignment).abs();
    let decision_matches = cpu_accepts == metal_accepts;
    let params_changed = update_delta_from_base > 0.0;
    let passed = params_changed
        && decision_matches
        && param_max_delta <= effective_tolerance
        && feature_max_delta <= effective_tolerance
        && loss_delta <= effective_tolerance
        && alignment_delta <= effective_tolerance;
    let rollout_launches = metal_train.rollout_kernel_launches;
    let state_update_launches = metal_train
        .state_update_kernel_launches
        .saturating_add(metal_train.hv_projection_kernel_launches);

    serde_json::json!({
        "ok": passed,
        "policy": "project_theseus_macos_metal_rollout_state_training_proof_v0",
        "state": if passed { "GREEN" } else { "RED" },
        "native_path": "rollout_state_update_plus_state_dynamics_training_update",
        "native_hot_loop": "rollout_state_update",
        "native_state_trainers": [
            "state_linear_update_kernel",
            "state_hv_projection_update_kernel"
        ],
        "parity_scope": "bounded_private_synthetic_rollout_state_training_update",
        "parity_for": "train-rollout-cuda-sweep state-training subpath",
        "cuda_equivalent": "rollout_cuda.rs::train_state_params_from_rollouts update_state_dynamics + update_hv_projection",
        "state_training_semantics_proof": true,
        "state_training_native_ported": passed,
        "cuda_state_training_parity_claim_allowed": false,
        "train_rollout_sweep_parity_claim_allowed": false,
        "full_cli_parity_claim_allowed": false,
        "production_scheduler_routing_enabled": false,
        "tolerance": config.tolerance,
        "effective_tolerance": effective_tolerance,
        "state_training": {
            "attempted": true,
            "state_epochs": config.state_epochs,
            "state_lr": config.state_lr,
            "cpu_candidate_accepts": cpu_accepts,
            "metal_candidate_accepts": metal_accepts,
            "decision_matches": decision_matches,
            "params_changed_from_base": params_changed,
            "param_update_max_abs_delta_from_base": update_delta_from_base
        },
        "metrics": {
            "param_max_abs_delta_cpu_vs_metal": param_max_delta,
            "feature_max_abs_delta_cpu_vs_metal": feature_max_delta,
            "loss_delta_cpu_vs_metal": loss_delta,
            "alignment_delta_cpu_vs_metal": alignment_delta,
            "base_target_mse": base_loss,
            "cpu_candidate_target_mse": cpu_candidate_loss,
            "metal_candidate_target_mse": metal_candidate_loss,
            "base_target_alignment": base_alignment,
            "cpu_candidate_target_alignment": cpu_candidate_alignment,
            "metal_candidate_target_alignment": metal_candidate_alignment
        },
        "param_deltas": param_deltas,
        "runtime_profile": {
            "backend": "apple_metal",
            "native_rust_owned": true,
            "python_mlx_bridge_used": false,
            "state_update_on_metal": true,
            "rollout_state_on_metal": true
        },
        "kernel_launches": rollout_launches.saturating_add(state_update_launches),
        "kernel_launch_breakdown": {
            "rollout_state_update": rollout_launches,
            "state_linear_update": metal_train.state_update_kernel_launches,
            "state_hv_projection_update": metal_train.hv_projection_kernel_launches
        },
        "config": {
            "cases": config.cases,
            "batch": config.batch,
            "steps": config.steps,
            "obs_dim": config.obs_dim,
            "hidden_dim": config.hidden_dim,
            "reservoir_dim": config.reservoir_dim,
            "hv_dim": config.hv_dim,
            "output_dim": config.output_dim,
            "dt": config.dt,
            "alpha": config.alpha,
            "memory_decay": config.memory_decay,
            "state_epochs": config.state_epochs,
            "state_lr": config.state_lr,
            "train_case_offset": config.train_case_offset,
            "eval_case_offset": config.eval_case_offset
        },
        "timing": {
            "cpu_state_training_wall_ms": cpu_train.wall_ms,
            "metal_state_training_wall_ms": metal_train.wall_ms,
            "report_wall_ms": started.elapsed().as_secs_f64() * 1000.0
        },
        "guardrails": {
            "no_public_calibration": true,
            "no_public_training_rows": true,
            "no_teacher": true,
            "no_external_inference": true,
            "no_fallback_returns": true,
            "does_not_claim_full_kernel_parity": true,
            "does_not_claim_full_training_parity": true,
            "does_not_route_scheduler_to_metal": true
        },
        "external_inference_calls": 0,
        "teacher_used": false,
        "public_training_rows": 0,
        "fallback_returns": 0,
        "model_promotion_allowed": false
    })
}

pub fn rollout_metal_train_path_proof_report(
    config: &RolloutMetalProofConfig,
) -> serde_json::Value {
    let started = std::time::Instant::now();
    if config.cases == 0
        || config.output_dim == 0
        || config.readout_epochs == 0
        || config.readout_lr <= 0.0
        || config.samples_per_launch == 0
    {
        return train_path_failed_report(
            started,
            config,
            "train path proof requires cases, output_dim, epochs, lr, and samples_per_launch",
        );
    }
    if !metal_available() {
        return serde_json::json!({
            "ok": false,
            "policy": "project_theseus_macos_metal_train_rollout_path_proof_v0",
            "state": "RED",
            "reason": "no_default_metal_device",
            "native_path": "train_rollout_style_feature_readout_train_eval_report",
            "parity_claim_allowed": false,
            "full_cli_parity_claim_allowed": false,
            "train_rollout_parity_claim_allowed": false,
            "external_inference_calls": 0,
            "teacher_used": false,
            "public_training_rows": 0
        });
    }

    let params = rollout_params(config);
    let cpu_train_features = match rollout_feature_tensor_with_offset(
        config,
        &params,
        rollout_backend_cpu,
        config.train_case_offset,
    ) {
        Ok(value) => value,
        Err(error) => {
            return train_path_failed_report(
                started,
                config,
                &format!("cpu_train_feature_path_failed:{error}"),
            )
        }
    };
    let metal_train_features = match rollout_feature_tensor_with_offset(
        config,
        &params,
        rollout_backend_metal,
        config.train_case_offset,
    ) {
        Ok(value) => value,
        Err(error) => {
            return train_path_failed_report(
                started,
                config,
                &format!("metal_train_feature_path_failed:{error}"),
            )
        }
    };
    let cpu_eval_features = match rollout_feature_tensor_with_offset(
        config,
        &params,
        rollout_backend_cpu,
        config.eval_case_offset,
    ) {
        Ok(value) => value,
        Err(error) => {
            return train_path_failed_report(
                started,
                config,
                &format!("cpu_eval_feature_path_failed:{error}"),
            )
        }
    };
    let metal_eval_features = match rollout_feature_tensor_with_offset(
        config,
        &params,
        rollout_backend_metal,
        config.eval_case_offset,
    ) {
        Ok(value) => value,
        Err(error) => {
            return train_path_failed_report(
                started,
                config,
                &format!("metal_eval_feature_path_failed:{error}"),
            )
        }
    };

    let train_targets = deterministic_targets(config.cases, config.output_dim);
    let eval_targets =
        deterministic_targets_with_offset(config.cases, config.output_dim, config.eval_case_offset);
    let mut cpu_readout = deterministic_readout(config);
    let mut metal_readout = cpu_readout.clone();

    let cpu_started = std::time::Instant::now();
    let cpu_train_trace = match train_readout_sgd_cpu_reference(
        &cpu_train_features,
        &train_targets,
        &mut cpu_readout,
        config.readout_epochs,
        config.readout_lr,
    ) {
        Ok(value) => value,
        Err(error) => {
            return train_path_failed_report(
                started,
                config,
                &format!("cpu_train_readout_failed:{error}"),
            )
        }
    };
    let cpu_eval_logits = match cpu_readout.logits(&cpu_eval_features) {
        Ok(value) => value,
        Err(error) => {
            return train_path_failed_report(
                started,
                config,
                &format!("cpu_eval_logits_failed:{error}"),
            )
        }
    };
    let cpu_eval_trace = match eval_trace_from_logits(&cpu_eval_logits, &eval_targets) {
        Ok(value) => value,
        Err(error) => {
            return train_path_failed_report(
                started,
                config,
                &format!("cpu_eval_trace_failed:{error}"),
            )
        }
    };
    let cpu_wall_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;

    let metal_started = std::time::Instant::now();
    let metal_train_trace = match train_readout_sgd_metal(
        &metal_train_features,
        &train_targets,
        &mut metal_readout,
        config.readout_epochs,
        config.readout_lr,
        config.samples_per_launch,
    ) {
        Ok(value) => value,
        Err(error) => {
            return train_path_failed_report(
                started,
                config,
                &format!("metal_train_readout_failed:{error}"),
            )
        }
    };
    let metal_eval_logits = match linear_readout_logits_metal(&metal_eval_features, &metal_readout)
    {
        Ok(value) => value,
        Err(error) => {
            return train_path_failed_report(
                started,
                config,
                &format!("metal_eval_logits_failed:{error}"),
            )
        }
    };
    let metal_eval_trace = match eval_trace_from_logits(&metal_eval_logits, &eval_targets) {
        Ok(value) => value,
        Err(error) => {
            return train_path_failed_report(
                started,
                config,
                &format!("metal_eval_trace_failed:{error}"),
            )
        }
    };
    let metal_wall_ms = metal_started.elapsed().as_secs_f64() * 1000.0;

    let train_feature_max_delta =
        max_abs_delta(&cpu_train_features.data, &metal_train_features.data);
    let eval_feature_max_delta = max_abs_delta(&cpu_eval_features.data, &metal_eval_features.data);
    let weight_max_delta = max_abs_delta(&cpu_readout.weights, &metal_readout.weights);
    let bias_max_delta = max_abs_delta(&cpu_readout.bias, &metal_readout.bias);
    let eval_logits_max_delta = max_abs_delta(&cpu_eval_logits.data, &metal_eval_logits.data);
    let eval_prediction_agreement = prediction_agreement(&cpu_eval_logits, &metal_eval_logits);
    let train_loss_delta = (cpu_train_trace.loss - metal_train_trace.loss).abs();
    let train_accuracy_delta = (cpu_train_trace.accuracy - metal_train_trace.accuracy).abs();
    let eval_loss_delta = (cpu_eval_trace.loss - metal_eval_trace.loss).abs();
    let eval_accuracy_delta = (cpu_eval_trace.accuracy - metal_eval_trace.accuracy).abs();
    let passed = train_feature_max_delta <= config.tolerance
        && eval_feature_max_delta <= config.tolerance
        && weight_max_delta <= config.tolerance
        && bias_max_delta <= config.tolerance
        && eval_logits_max_delta <= config.tolerance
        && train_loss_delta <= config.tolerance
        && train_accuracy_delta <= f32::EPSILON
        && eval_loss_delta <= config.tolerance
        && eval_accuracy_delta <= f32::EPSILON
        && (eval_prediction_agreement - 1.0).abs() <= f32::EPSILON;

    let train_rollout_launches = config.cases.div_ceil(config.batch) * config.steps;
    let eval_rollout_launches = config.cases.div_ceil(config.batch) * config.steps;
    let readout_train_launches =
        config.readout_epochs * config.cases.div_ceil(config.samples_per_launch.max(1));
    let readout_eval_launches = config.cases.saturating_mul(config.output_dim);
    let kernel_launches = train_rollout_launches
        + eval_rollout_launches
        + readout_train_launches
        + readout_eval_launches;
    let train_examples_per_second = (config.cases.saturating_mul(config.readout_epochs)) as f64
        / (metal_wall_ms.max(1.0) / 1000.0);
    let artifact_write = if let Some(path) = config
        .artifact_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let labels = rollout_label_names(config.output_dim);
        match symliquid_core::benchmarks::write_readout_artifact(
            path,
            &metal_readout,
            &labels,
            config.hv_dim,
            "metal_rollout_memory_readout_sgd_private_synthetic_train_eval",
        ) {
            Ok(()) => serde_json::json!({
                "attempted": true,
                "path": path,
                "kind": "canonical_readout_artifact",
                "schema": "symliquid_core::benchmarks::ReadoutArtifact",
                "production_checkpoint_compatible": true,
                "promotion_allowed": false,
                "weights_written": metal_readout.weights.len(),
                "bias_written": metal_readout.bias.len(),
                "labels_written": labels.len(),
                "reason": "Canonical readout weights/bias artifact written for Metal train-rollout path; scheduler routing remains separately gated."
            }),
            Err(error) => {
                return train_path_failed_report(
                    started,
                    config,
                    &format!("artifact_write_failed:{error}"),
                )
            }
        }
    } else {
        serde_json::json!({
            "attempted": false,
            "reason": "no_artifact_path_requested"
        })
    };

    serde_json::json!({
        "ok": passed,
        "policy": "project_theseus_macos_metal_train_rollout_path_proof_v0",
        "state": if passed { "GREEN" } else { "RED" },
        "native_path": "train_rollout_style_feature_readout_train_eval_report",
        "native_hot_loop": "rollout_state_update",
        "native_trainer": "readout_sgd_samples_kernel",
        "native_readout": "linear_readout_logits_kernel",
        "parity_scope": "bounded_private_synthetic_train_eval_rollout_features_readout_training_report",
        "parity_claim_allowed": false,
        "full_cli_parity_claim_allowed": false,
        "train_rollout_parity_claim_allowed": false,
        "cuda_equivalent": "train-rollout-cuda feature/readout/eval subpath",
        "model_id": "symliquid-metal-train-rollout-path-proof",
        "feature_set": "metal_rollout_memory_readout_sgd_private_synthetic_train_eval",
        "train_rows": config.cases,
        "eval_rows": config.cases,
        "epochs": config.readout_epochs,
        "hv_dim": config.hv_dim,
        "labels": config.output_dim,
        "symbolic_fallback": false,
        "artifact_write": artifact_write,
        "train_metrics": {
            "cpu_loss": cpu_train_trace.loss,
            "metal_loss": metal_train_trace.loss,
            "loss_delta": train_loss_delta,
            "cpu_accuracy": cpu_train_trace.accuracy,
            "metal_accuracy": metal_train_trace.accuracy,
            "accuracy_delta": train_accuracy_delta
        },
        "eval_metrics": {
            "cpu_loss": cpu_eval_trace.loss,
            "metal_loss": metal_eval_trace.loss,
            "loss_delta": eval_loss_delta,
            "cpu_accuracy": cpu_eval_trace.accuracy,
            "metal_accuracy": metal_eval_trace.accuracy,
            "accuracy_delta": eval_accuracy_delta,
            "prediction_agreement": eval_prediction_agreement
        },
        "parity_metrics": {
            "train_feature_max_abs_delta": train_feature_max_delta,
            "eval_feature_max_abs_delta": eval_feature_max_delta,
            "weight_max_abs_delta": weight_max_delta,
            "bias_max_abs_delta": bias_max_delta,
            "eval_logits_max_abs_delta": eval_logits_max_delta
        },
        "runtime_profile": {
            "backend": "apple_metal",
            "native_rust_owned": true,
            "python_mlx_bridge_used": false
        },
        "timing": {
            "cpu_wall_ms": cpu_wall_ms,
            "metal_wall_ms": metal_wall_ms,
            "report_wall_ms": started.elapsed().as_secs_f64() * 1000.0,
            "metal_train_examples_per_second": train_examples_per_second
        },
        "kernel_launches": kernel_launches,
        "kernel_launch_breakdown": {
            "train_rollout": train_rollout_launches,
            "eval_rollout": eval_rollout_launches,
            "readout_train": readout_train_launches,
            "readout_eval_logit_groups": readout_eval_launches
        },
        "config": {
            "cases": config.cases,
            "batch": config.batch,
            "steps": config.steps,
            "obs_dim": config.obs_dim,
            "hidden_dim": config.hidden_dim,
            "reservoir_dim": config.reservoir_dim,
            "hv_dim": config.hv_dim,
            "output_dim": config.output_dim,
            "dt": config.dt,
            "alpha": config.alpha,
            "memory_decay": config.memory_decay,
            "readout_epochs": config.readout_epochs,
            "readout_lr": config.readout_lr,
            "samples_per_launch": config.samples_per_launch,
            "train_case_offset": config.train_case_offset,
            "eval_case_offset": config.eval_case_offset,
            "artifact_path": config.artifact_path
        },
        "guardrails": {
            "no_public_calibration": true,
            "no_public_training_rows": true,
            "no_teacher": true,
            "no_external_inference": true,
            "no_fallback_returns": true,
            "does_not_claim_full_kernel_parity": true,
            "does_not_claim_full_training_parity": true,
            "does_not_route_scheduler_to_metal": true
        },
        "external_inference_calls": 0,
        "teacher_used": false,
        "public_training_rows": 0,
        "model_promotion_allowed": false
    })
}

pub fn rollout_metal_proof_report(config: &RolloutMetalProofConfig) -> serde_json::Value {
    let started = std::time::Instant::now();
    let rollout_config = symliquid_cuda::rollout_cuda::RolloutConfig {
        batch: config.batch,
        obs_dim: config.obs_dim,
        hidden_dim: config.hidden_dim,
        reservoir_dim: config.reservoir_dim,
        hv_dim: config.hv_dim,
        dt: config.dt,
        alpha: config.alpha,
        memory_decay: config.memory_decay,
    };
    let observations = match symliquid_core::tensor::Tensor::new(
        config.batch.saturating_mul(config.steps),
        config.obs_dim,
        deterministic_values(
            config
                .batch
                .saturating_mul(config.steps)
                .saturating_mul(config.obs_dim),
            0.07,
        ),
    ) {
        Ok(value) => value,
        Err(error) => return failed_report(started, config, "observation_tensor_failed", error),
    };
    let state = match rollout_state(config) {
        Ok(value) => value,
        Err(error) => return failed_report(started, config, "state_tensor_failed", error),
    };
    let params = rollout_params(config);

    if !metal_available() {
        return serde_json::json!({
            "ok": false,
            "policy": "project_theseus_macos_metal_rollout_hot_loop_proof_v0",
            "state": "RED",
            "reason": "no_default_metal_device",
            "native_hot_loop": "rollout_state_update",
            "parity_claim_allowed": false,
            "full_cli_parity_claim_allowed": false,
            "external_inference_calls": 0,
            "teacher_used": false,
            "public_training_rows": 0
        });
    }

    let cpu_started = std::time::Instant::now();
    let cpu = match symliquid_cuda::rollout_cuda::rollout_sequence_cpu(
        &observations,
        &state,
        &params,
        &rollout_config,
    ) {
        Ok(value) => value,
        Err(error) => return failed_report(started, config, "cpu_reference_failed", error),
    };
    let cpu_wall_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;

    let metal_started = std::time::Instant::now();
    let metal = match rollout_sequence_metal(&observations, &state, &params, &rollout_config) {
        Ok(value) => value,
        Err(error) => return failed_report(started, config, "metal_native_failed", error),
    };
    let metal_wall_ms = metal_started.elapsed().as_secs_f64() * 1000.0;

    let h_delta = max_abs_delta(&cpu.h.data, &metal.h.data);
    let r_delta = max_abs_delta(&cpu.r.data, &metal.r.data);
    let memory_delta = max_abs_delta(&cpu.memory.data, &metal.memory.data);
    let max_delta = h_delta.max(r_delta).max(memory_delta);
    let passed = max_delta <= config.tolerance;

    serde_json::json!({
        "ok": passed,
        "policy": "project_theseus_macos_metal_rollout_hot_loop_proof_v0",
        "state": if passed { "GREEN" } else { "RED" },
        "native_hot_loop": "rollout_state_update",
        "parity_scope": "single_kernel_reference_case",
        "parity_claim_allowed": false,
        "full_cli_parity_claim_allowed": false,
        "cuda_equivalent": "crates/symliquid-cuda/kernels/rollout_kernels.cu::rollout_state_update_kernel",
        "metal_source": "crates/symliquid-metal/kernels/rollout_state_update.metal",
        "rust_owner": "crates/symliquid-metal",
        "tolerance": config.tolerance,
        "max_abs_delta": max_delta,
        "component_max_abs_delta": {
            "h": h_delta,
            "r": r_delta,
            "memory": memory_delta
        },
        "config": {
            "batch": config.batch,
            "obs_dim": config.obs_dim,
            "hidden_dim": config.hidden_dim,
            "reservoir_dim": config.reservoir_dim,
            "hv_dim": config.hv_dim,
            "steps": config.steps,
            "dt": config.dt,
            "alpha": config.alpha,
            "memory_decay": config.memory_decay
        },
        "timing": {
            "cpu_wall_ms": cpu_wall_ms,
            "metal_wall_ms": metal_wall_ms,
            "report_wall_ms": started.elapsed().as_secs_f64() * 1000.0
        },
        "guardrails": {
            "no_public_calibration": true,
            "no_public_training_rows": true,
            "no_teacher": true,
            "no_external_inference": true,
            "does_not_claim_full_kernel_parity": true,
            "does_not_route_scheduler_to_metal": true
        },
        "external_inference_calls": 0,
        "teacher_used": false,
        "public_training_rows": 0,
        "model_promotion_allowed": false
    })
}

fn rollout_feature_tensor(
    config: &RolloutMetalFeatureProofConfig,
    params: &symliquid_cuda::rollout_cuda::RolloutParams,
    backend: fn(
        &symliquid_core::tensor::Tensor,
        &symliquid_cuda::rollout_cuda::RolloutState,
        &symliquid_cuda::rollout_cuda::RolloutParams,
        &symliquid_cuda::rollout_cuda::RolloutConfig,
    ) -> symliquid_core::error::Result<symliquid_cuda::rollout_cuda::RolloutState>,
) -> symliquid_core::error::Result<symliquid_core::tensor::Tensor> {
    rollout_feature_tensor_with_offset(config, params, backend, 0)
}

fn rollout_feature_tensor_with_offset(
    config: &RolloutMetalFeatureProofConfig,
    params: &symliquid_cuda::rollout_cuda::RolloutParams,
    backend: fn(
        &symliquid_core::tensor::Tensor,
        &symliquid_cuda::rollout_cuda::RolloutState,
        &symliquid_cuda::rollout_cuda::RolloutParams,
        &symliquid_cuda::rollout_cuda::RolloutConfig,
    ) -> symliquid_core::error::Result<symliquid_cuda::rollout_cuda::RolloutState>,
    case_offset: usize,
) -> symliquid_core::error::Result<symliquid_core::tensor::Tensor> {
    let mut features = Vec::with_capacity(config.cases.saturating_mul(config.hv_dim));
    for start in (0..config.cases).step_by(config.batch) {
        let rows = (config.cases - start).min(config.batch);
        let rollout = symliquid_cuda::rollout_cuda::RolloutConfig {
            batch: rows,
            obs_dim: config.obs_dim,
            hidden_dim: config.hidden_dim,
            reservoir_dim: config.reservoir_dim,
            hv_dim: config.hv_dim,
            dt: config.dt,
            alpha: config.alpha,
            memory_decay: config.memory_decay,
        };
        let observations =
            deterministic_observations(case_offset.saturating_add(start), rows, config)?;
        let state = zero_state(rows, config);
        let final_state = backend(&observations, &state, params, &rollout)?;
        for row in 0..rows {
            features.extend_from_slice(final_state.memory.row(row));
        }
    }
    symliquid_core::tensor::Tensor::new(config.cases, config.hv_dim, features)
}

fn rollout_backend_cpu(
    observations: &symliquid_core::tensor::Tensor,
    state: &symliquid_cuda::rollout_cuda::RolloutState,
    params: &symliquid_cuda::rollout_cuda::RolloutParams,
    config: &symliquid_cuda::rollout_cuda::RolloutConfig,
) -> symliquid_core::error::Result<symliquid_cuda::rollout_cuda::RolloutState> {
    symliquid_cuda::rollout_cuda::rollout_sequence_cpu(observations, state, params, config)
}

fn rollout_backend_metal(
    observations: &symliquid_core::tensor::Tensor,
    state: &symliquid_cuda::rollout_cuda::RolloutState,
    params: &symliquid_cuda::rollout_cuda::RolloutParams,
    config: &symliquid_cuda::rollout_cuda::RolloutConfig,
) -> symliquid_core::error::Result<symliquid_cuda::rollout_cuda::RolloutState> {
    rollout_sequence_metal(observations, state, params, config)
}

fn deterministic_observations(
    start_case: usize,
    cases: usize,
    config: &RolloutMetalFeatureProofConfig,
) -> symliquid_core::error::Result<symliquid_core::tensor::Tensor> {
    let mut data = Vec::with_capacity(
        cases
            .saturating_mul(config.steps)
            .saturating_mul(config.obs_dim),
    );
    for step in 0..config.steps {
        for row in 0..cases {
            let case_id = start_case + row;
            let mut obs = Vec::with_capacity(config.obs_dim);
            for col in 0..config.obs_dim {
                let hashed = ((case_id + 1) * 37 + (step + 3) * 17 + (col + 5) * 11) % 31;
                let phase = ((case_id ^ (step * 3) ^ (col * 5)) % 7) as f32 - 3.0;
                obs.push((hashed as f32 - 15.0) * 0.025 + phase * 0.01);
            }
            normalize_dense(&mut obs);
            data.extend_from_slice(&obs);
        }
    }
    symliquid_core::tensor::Tensor::new(cases.saturating_mul(config.steps), config.obs_dim, data)
}

fn zero_state(
    batch: usize,
    config: &RolloutMetalFeatureProofConfig,
) -> symliquid_cuda::rollout_cuda::RolloutState {
    symliquid_cuda::rollout_cuda::RolloutState {
        h: symliquid_core::tensor::Tensor::zeros(batch, config.hidden_dim),
        r: symliquid_core::tensor::Tensor::zeros(batch, config.reservoir_dim),
        memory: symliquid_core::tensor::Tensor::zeros(batch, config.hv_dim),
    }
}

fn train_state_params_for_proof(
    config: &RolloutMetalFeatureProofConfig,
    params: &mut symliquid_cuda::rollout_cuda::RolloutParams,
    rollout_backend: fn(
        &symliquid_core::tensor::Tensor,
        &symliquid_cuda::rollout_cuda::RolloutState,
        &symliquid_cuda::rollout_cuda::RolloutParams,
        &symliquid_cuda::rollout_cuda::RolloutConfig,
    ) -> symliquid_core::error::Result<
        symliquid_cuda::rollout_cuda::RolloutState,
    >,
    update_backend: StateTrainingUpdateBackend,
) -> symliquid_core::error::Result<StateTrainingProofTrace> {
    let started = std::time::Instant::now();
    let hidden_targets = state_label_vectors(config.output_dim, config.hidden_dim, 0x4849);
    let reservoir_targets = state_label_vectors(config.output_dim, config.reservoir_dim, 0x5253);
    let hv_targets = state_label_vectors(config.output_dim, config.hv_dim, 0x4856);
    let mut rollout_kernel_launches = 0usize;
    let mut state_update_kernel_launches = 0usize;
    let mut hv_projection_kernel_launches = 0usize;

    for _epoch in 0..config.state_epochs {
        for start in (0..config.cases).step_by(config.batch.max(1)) {
            let rows = (config.cases - start).min(config.batch.max(1));
            let rollout = symliquid_cuda::rollout_cuda::RolloutConfig {
                batch: rows,
                obs_dim: config.obs_dim,
                hidden_dim: config.hidden_dim,
                reservoir_dim: config.reservoir_dim,
                hv_dim: config.hv_dim,
                dt: config.dt,
                alpha: config.alpha,
                memory_decay: config.memory_decay,
            };
            let case_offset = config.train_case_offset.saturating_add(start);
            let observations = deterministic_observations(case_offset, rows, config)?;
            let state = zero_state(rows, config);
            let final_state = rollout_backend(&observations, &state, params, &rollout)?;
            rollout_kernel_launches = rollout_kernel_launches.saturating_add(config.steps);
            let targets = deterministic_targets_with_offset(rows, config.output_dim, case_offset);
            for (row, target) in targets.iter().copied().enumerate() {
                let obs_features =
                    pooled_observation_features_for_case(case_offset.saturating_add(row), config)?;
                update_state_dynamics_for_proof(
                    update_backend,
                    &obs_features,
                    final_state.h.row(row),
                    final_state.r.row(row),
                    &hidden_targets[target],
                    &reservoir_targets[target],
                    params,
                    &rollout,
                    config.state_lr * 0.15,
                )?;
                update_hv_projection_for_proof(
                    update_backend,
                    final_state.r.row(row),
                    &hv_targets[target],
                    params,
                    config.reservoir_dim,
                    config.state_lr,
                )?;
                if matches!(update_backend, StateTrainingUpdateBackend::MetalKernels) {
                    state_update_kernel_launches = state_update_kernel_launches.saturating_add(4);
                    hv_projection_kernel_launches = hv_projection_kernel_launches.saturating_add(1);
                }
            }
        }
    }

    Ok(StateTrainingProofTrace {
        wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        rollout_kernel_launches,
        state_update_kernel_launches,
        hv_projection_kernel_launches,
    })
}

#[allow(clippy::too_many_arguments)]
fn update_state_dynamics_for_proof(
    backend: StateTrainingUpdateBackend,
    obs_features: &[f32],
    hidden_state: &[f32],
    reservoir_state: &[f32],
    hidden_target: &[f32],
    reservoir_target: &[f32],
    params: &mut symliquid_cuda::rollout_cuda::RolloutParams,
    config: &symliquid_cuda::rollout_cuda::RolloutConfig,
    lr: f32,
) -> symliquid_core::error::Result<()> {
    match backend {
        StateTrainingUpdateBackend::CpuReference => {
            update_state_dynamics_cpu_reference(
                obs_features,
                hidden_state,
                reservoir_state,
                hidden_target,
                reservoir_target,
                params,
                config,
                lr,
            );
            Ok(())
        }
        StateTrainingUpdateBackend::MetalKernels => update_state_dynamics_sample_metal(
            obs_features,
            hidden_state,
            reservoir_state,
            hidden_target,
            reservoir_target,
            params,
            config,
            lr,
        ),
    }
}

fn update_hv_projection_for_proof(
    backend: StateTrainingUpdateBackend,
    reservoir_state: &[f32],
    target_hv: &[f32],
    params: &mut symliquid_cuda::rollout_cuda::RolloutParams,
    reservoir_dim: usize,
    lr: f32,
) -> symliquid_core::error::Result<()> {
    match backend {
        StateTrainingUpdateBackend::CpuReference => {
            update_hv_projection_cpu_reference(
                reservoir_state,
                target_hv,
                params,
                reservoir_dim,
                lr,
            );
            Ok(())
        }
        StateTrainingUpdateBackend::MetalKernels => {
            update_hv_projection_metal(reservoir_state, target_hv, params, reservoir_dim, lr)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn update_state_dynamics_cpu_reference(
    obs_features: &[f32],
    hidden_state: &[f32],
    reservoir_state: &[f32],
    hidden_target: &[f32],
    reservoir_target: &[f32],
    params: &mut symliquid_cuda::rollout_cuda::RolloutParams,
    config: &symliquid_cuda::rollout_cuda::RolloutConfig,
    lr: f32,
) {
    update_linear_state_block_cpu_reference(StateLinearBlockUpdate {
        weights: &mut params.obs_to_h,
        bias: Some(&mut params.h_bias),
        rows: config.hidden_dim,
        cols: config.obs_dim,
        input: obs_features,
        output: hidden_state,
        target: hidden_target,
        lr,
        max_abs: 0.8,
    });
    update_linear_state_block_cpu_reference(StateLinearBlockUpdate {
        weights: &mut params.h_recurrent,
        bias: None,
        rows: config.hidden_dim,
        cols: config.hidden_dim,
        input: hidden_state,
        output: hidden_state,
        target: hidden_target,
        lr: lr * 0.25,
        max_abs: 0.25,
    });
    update_linear_state_block_cpu_reference(StateLinearBlockUpdate {
        weights: &mut params.reservoir_input,
        bias: Some(&mut params.reservoir_bias),
        rows: config.reservoir_dim,
        cols: config.hidden_dim,
        input: hidden_state,
        output: reservoir_state,
        target: reservoir_target,
        lr,
        max_abs: 0.8,
    });
    update_linear_state_block_cpu_reference(StateLinearBlockUpdate {
        weights: &mut params.reservoir_recurrent,
        bias: None,
        rows: config.reservoir_dim,
        cols: config.reservoir_dim,
        input: reservoir_state,
        output: reservoir_state,
        target: reservoir_target,
        lr: lr * 0.2,
        max_abs: 0.2,
    });
}

struct StateLinearBlockUpdate<'a> {
    weights: &'a mut [f32],
    bias: Option<&'a mut [f32]>,
    rows: usize,
    cols: usize,
    input: &'a [f32],
    output: &'a [f32],
    target: &'a [f32],
    lr: f32,
    max_abs: f32,
}

fn update_linear_state_block_cpu_reference(mut update: StateLinearBlockUpdate<'_>) {
    let norm = update
        .input
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .max(1.0e-6);
    for row in 0..update.rows {
        let err = (update.target[row] - update.output[row].tanh()).clamp(-2.0, 2.0);
        let scale = update.lr * err / norm;
        let row_start = row * update.cols;
        for (weight, value) in update.weights[row_start..row_start + update.cols]
            .iter_mut()
            .zip(update.input)
        {
            *weight = (*weight + scale * value).clamp(-update.max_abs, update.max_abs);
        }
        if let Some(bias_values) = update.bias.as_deref_mut() {
            bias_values[row] = (bias_values[row] + update.lr * 0.1 * err).clamp(-0.5, 0.5);
        }
    }
}

fn update_hv_projection_cpu_reference(
    reservoir_state: &[f32],
    target_hv: &[f32],
    params: &mut symliquid_cuda::rollout_cuda::RolloutParams,
    reservoir_dim: usize,
    lr: f32,
) {
    let norm = reservoir_state
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .max(1.0e-6);
    for (hv_idx, target) in target_hv.iter().copied().enumerate() {
        let row_start = hv_idx * reservoir_dim;
        let row = &mut params.hv_proj[row_start..row_start + reservoir_dim];
        let dot = row
            .iter()
            .zip(reservoir_state)
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
        let pred = if dot >= 0.0 { 1.0 } else { -1.0 };
        let err = target - pred;
        if err.abs() > f32::EPSILON {
            let scale = lr * err / norm;
            for (weight, value) in row.iter_mut().zip(reservoir_state) {
                *weight = (*weight + scale * value).clamp(-1.0, 1.0);
            }
        }
    }
}

fn pooled_observation_features_for_case(
    case_id: usize,
    config: &RolloutMetalFeatureProofConfig,
) -> symliquid_core::error::Result<Vec<f32>> {
    let observations = deterministic_observations(case_id, 1, config)?;
    let mut pooled = vec![0.0; config.obs_dim];
    let scale = 1.0 / config.steps.max(1) as f32;
    for step in 0..config.steps {
        let row = observations.row(step);
        for (dst, value) in pooled.iter_mut().zip(row) {
            *dst += value * scale;
        }
    }
    normalize_dense(&mut pooled);
    Ok(pooled)
}

fn state_label_vectors(labels: usize, dim: usize, namespace: u64) -> Vec<Vec<f32>> {
    (0..labels)
        .map(|label| {
            (0..dim)
                .map(|idx| {
                    let mut hash = namespace
                        ^ ((label as u64 + 1).wrapping_mul(0x9E37_79B1_85EB_CA87))
                        ^ ((idx as u64 + 11).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
                    hash ^= hash >> 33;
                    hash = hash.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
                    hash ^= hash >> 33;
                    if hash & 1 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect()
        })
        .collect()
}

fn mean_target_mse(
    features: &symliquid_core::tensor::Tensor,
    targets: &[usize],
    label_vectors: &[Vec<f32>],
) -> f32 {
    if features.rows == 0 || features.cols == 0 || targets.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for row in 0..features.rows.min(targets.len()) {
        if let Some(target) = label_vectors.get(targets[row]) {
            let row_values = features.row(row);
            total += row_values
                .iter()
                .zip(target)
                .map(|(value, target)| {
                    let diff = value - target;
                    diff * diff
                })
                .sum::<f32>()
                / row_values.len().max(1) as f32;
            count += 1;
        }
    }
    total / count.max(1) as f32
}

fn mean_target_alignment(
    features: &symliquid_core::tensor::Tensor,
    targets: &[usize],
    label_vectors: &[Vec<f32>],
) -> f32 {
    if features.rows == 0 || features.cols == 0 || targets.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for row in 0..features.rows.min(targets.len()) {
        if let Some(target) = label_vectors.get(targets[row]) {
            total += features
                .row(row)
                .iter()
                .zip(target)
                .map(|(value, target)| value * target)
                .sum::<f32>()
                / features.cols.max(1) as f32;
            count += 1;
        }
    }
    total / count.max(1) as f32
}

fn state_training_candidate_accepts(base_loss: f32, candidate_loss: f32) -> bool {
    candidate_loss.is_finite() && candidate_loss <= base_loss * 1.01
}

fn rollout_param_deltas(
    left: &symliquid_cuda::rollout_cuda::RolloutParams,
    right: &symliquid_cuda::rollout_cuda::RolloutParams,
) -> BTreeMap<String, f32> {
    BTreeMap::from([
        (
            "obs_to_h".to_string(),
            max_abs_delta(&left.obs_to_h, &right.obs_to_h),
        ),
        (
            "h_recurrent".to_string(),
            max_abs_delta(&left.h_recurrent, &right.h_recurrent),
        ),
        (
            "h_bias".to_string(),
            max_abs_delta(&left.h_bias, &right.h_bias),
        ),
        (
            "reservoir_input".to_string(),
            max_abs_delta(&left.reservoir_input, &right.reservoir_input),
        ),
        (
            "reservoir_recurrent".to_string(),
            max_abs_delta(&left.reservoir_recurrent, &right.reservoir_recurrent),
        ),
        (
            "reservoir_bias".to_string(),
            max_abs_delta(&left.reservoir_bias, &right.reservoir_bias),
        ),
        (
            "hv_proj".to_string(),
            max_abs_delta(&left.hv_proj, &right.hv_proj),
        ),
    ])
}

fn rollout_param_max_delta(
    left: &symliquid_cuda::rollout_cuda::RolloutParams,
    right: &symliquid_cuda::rollout_cuda::RolloutParams,
) -> f32 {
    rollout_param_deltas(left, right)
        .values()
        .copied()
        .fold(0.0_f32, f32::max)
}

fn feature_failed_report(
    started: std::time::Instant,
    config: &RolloutMetalFeatureProofConfig,
    reason: &str,
) -> serde_json::Value {
    serde_json::json!({
        "ok": false,
        "policy": "project_theseus_macos_metal_rollout_feature_path_proof_v0",
        "state": "RED",
        "reason": reason,
        "native_path": "rollout_memory_feature_tensor",
        "parity_scope": "bounded_private_synthetic_rollout_feature_batch",
        "parity_claim_allowed": false,
        "full_cli_parity_claim_allowed": false,
        "train_rollout_parity_claim_allowed": false,
        "config": {
            "cases": config.cases,
            "batch": config.batch,
            "obs_dim": config.obs_dim,
            "hidden_dim": config.hidden_dim,
            "reservoir_dim": config.reservoir_dim,
            "hv_dim": config.hv_dim,
            "steps": config.steps,
            "dt": config.dt,
            "alpha": config.alpha,
            "memory_decay": config.memory_decay
        },
        "timing": {
            "report_wall_ms": started.elapsed().as_secs_f64() * 1000.0
        },
        "external_inference_calls": 0,
        "teacher_used": false,
        "public_training_rows": 0,
        "model_promotion_allowed": false
    })
}

fn failed_report(
    started: std::time::Instant,
    config: &RolloutMetalProofConfig,
    reason: &str,
    error: symliquid_core::error::SymError,
) -> serde_json::Value {
    serde_json::json!({
        "ok": false,
        "policy": "project_theseus_macos_metal_rollout_hot_loop_proof_v0",
        "state": "RED",
        "reason": format!("{reason}:{error}"),
        "native_hot_loop": "rollout_state_update",
        "parity_scope": "single_kernel_reference_case",
        "parity_claim_allowed": false,
        "full_cli_parity_claim_allowed": false,
        "config": {
            "batch": config.batch,
            "obs_dim": config.obs_dim,
            "hidden_dim": config.hidden_dim,
            "reservoir_dim": config.reservoir_dim,
            "hv_dim": config.hv_dim,
            "steps": config.steps,
            "dt": config.dt,
            "alpha": config.alpha,
            "memory_decay": config.memory_decay
        },
        "timing": {
            "report_wall_ms": started.elapsed().as_secs_f64() * 1000.0
        },
        "external_inference_calls": 0,
        "teacher_used": false,
        "public_training_rows": 0,
        "model_promotion_allowed": false
    })
}

fn readout_failed_report(
    started: std::time::Instant,
    config: &RolloutMetalProofConfig,
    reason: &str,
) -> serde_json::Value {
    serde_json::json!({
        "ok": false,
        "policy": "project_theseus_macos_metal_rollout_readout_proof_v0",
        "state": "RED",
        "reason": reason,
        "native_path": "rollout_memory_feature_tensor_plus_linear_readout_logits",
        "parity_scope": "bounded_private_synthetic_rollout_feature_batch_plus_readout_logits",
        "parity_claim_allowed": false,
        "full_cli_parity_claim_allowed": false,
        "train_rollout_parity_claim_allowed": false,
        "config": {
            "cases": config.cases,
            "batch": config.batch,
            "obs_dim": config.obs_dim,
            "hidden_dim": config.hidden_dim,
            "reservoir_dim": config.reservoir_dim,
            "hv_dim": config.hv_dim,
            "output_dim": config.output_dim,
            "steps": config.steps,
            "dt": config.dt,
            "alpha": config.alpha,
            "memory_decay": config.memory_decay
        },
        "timing": {
            "report_wall_ms": started.elapsed().as_secs_f64() * 1000.0
        },
        "external_inference_calls": 0,
        "teacher_used": false,
        "public_training_rows": 0,
        "model_promotion_allowed": false
    })
}

fn readout_training_failed_report(
    started: std::time::Instant,
    config: &RolloutMetalProofConfig,
    reason: &str,
) -> serde_json::Value {
    serde_json::json!({
        "ok": false,
        "policy": "project_theseus_macos_metal_rollout_readout_training_proof_v0",
        "state": "RED",
        "reason": reason,
        "native_path": "rollout_memory_feature_tensor_plus_readout_sgd_training_plus_logits_eval",
        "parity_scope": "bounded_private_synthetic_rollout_feature_batch_plus_readout_sgd_training_and_eval",
        "parity_claim_allowed": false,
        "full_cli_parity_claim_allowed": false,
        "train_rollout_parity_claim_allowed": false,
        "config": {
            "cases": config.cases,
            "batch": config.batch,
            "obs_dim": config.obs_dim,
            "hidden_dim": config.hidden_dim,
            "reservoir_dim": config.reservoir_dim,
            "hv_dim": config.hv_dim,
            "output_dim": config.output_dim,
            "steps": config.steps,
            "dt": config.dt,
            "alpha": config.alpha,
            "memory_decay": config.memory_decay,
            "readout_epochs": config.readout_epochs,
            "readout_lr": config.readout_lr,
            "samples_per_launch": config.samples_per_launch
        },
        "timing": {
            "report_wall_ms": started.elapsed().as_secs_f64() * 1000.0
        },
        "external_inference_calls": 0,
        "teacher_used": false,
        "public_training_rows": 0,
        "model_promotion_allowed": false
    })
}

fn state_training_failed_report(
    started: std::time::Instant,
    config: &RolloutMetalProofConfig,
    reason: &str,
) -> serde_json::Value {
    serde_json::json!({
        "ok": false,
        "policy": "project_theseus_macos_metal_rollout_state_training_proof_v0",
        "state": "RED",
        "reason": reason,
        "native_path": "rollout_state_update_plus_state_dynamics_training_update",
        "parity_scope": "bounded_private_synthetic_rollout_state_training_update",
        "state_training_semantics_proof": false,
        "state_training_native_ported": false,
        "cuda_state_training_parity_claim_allowed": false,
        "train_rollout_sweep_parity_claim_allowed": false,
        "full_cli_parity_claim_allowed": false,
        "production_scheduler_routing_enabled": false,
        "config": {
            "cases": config.cases,
            "batch": config.batch,
            "obs_dim": config.obs_dim,
            "hidden_dim": config.hidden_dim,
            "reservoir_dim": config.reservoir_dim,
            "hv_dim": config.hv_dim,
            "output_dim": config.output_dim,
            "steps": config.steps,
            "dt": config.dt,
            "alpha": config.alpha,
            "memory_decay": config.memory_decay,
            "state_epochs": config.state_epochs,
            "state_lr": config.state_lr,
            "train_case_offset": config.train_case_offset,
            "eval_case_offset": config.eval_case_offset
        },
        "guardrails": {
            "no_public_calibration": true,
            "no_public_training_rows": true,
            "no_teacher": true,
            "no_external_inference": true,
            "no_fallback_returns": true,
            "does_not_claim_full_kernel_parity": true,
            "does_not_claim_full_training_parity": true,
            "does_not_route_scheduler_to_metal": true
        },
        "timing": {
            "report_wall_ms": started.elapsed().as_secs_f64() * 1000.0
        },
        "external_inference_calls": 0,
        "teacher_used": false,
        "public_training_rows": 0,
        "fallback_returns": 0,
        "model_promotion_allowed": false
    })
}

fn train_path_failed_report(
    started: std::time::Instant,
    config: &RolloutMetalProofConfig,
    reason: &str,
) -> serde_json::Value {
    serde_json::json!({
        "ok": false,
        "policy": "project_theseus_macos_metal_train_rollout_path_proof_v0",
        "state": "RED",
        "reason": reason,
        "native_path": "train_rollout_style_feature_readout_train_eval_report",
        "parity_scope": "bounded_private_synthetic_train_eval_rollout_features_readout_training_report",
        "parity_claim_allowed": false,
        "full_cli_parity_claim_allowed": false,
        "train_rollout_parity_claim_allowed": false,
        "config": {
            "cases": config.cases,
            "batch": config.batch,
            "obs_dim": config.obs_dim,
            "hidden_dim": config.hidden_dim,
            "reservoir_dim": config.reservoir_dim,
            "hv_dim": config.hv_dim,
            "output_dim": config.output_dim,
            "steps": config.steps,
            "dt": config.dt,
            "alpha": config.alpha,
            "memory_decay": config.memory_decay,
            "readout_epochs": config.readout_epochs,
            "readout_lr": config.readout_lr,
            "samples_per_launch": config.samples_per_launch
        },
        "timing": {
            "report_wall_ms": started.elapsed().as_secs_f64() * 1000.0
        },
        "external_inference_calls": 0,
        "teacher_used": false,
        "public_training_rows": 0,
        "model_promotion_allowed": false
    })
}

fn token_superposition_failed_report(
    started: std::time::Instant,
    config: &TokenSuperpositionMetalProofConfig,
    reason: &str,
) -> serde_json::Value {
    serde_json::json!({
        "ok": false,
        "policy": "project_theseus_macos_metal_token_superposition_readout_proof_v0",
        "state": "RED",
        "reason": reason,
        "native_path": "token_superposition_bag_readout_sgd_plus_ar_recovery",
        "parity_scope": "bounded_private_synthetic_token_superposition_readout",
        "train_token_superposition_parity_claim_allowed": false,
        "full_cli_parity_claim_allowed": false,
        "model_promotion_allowed": false,
        "config": {
            "vocab_size": config.vocab_size,
            "hv_dim": config.hv_dim,
            "train_tokens": config.train_tokens,
            "train_samples": config.train_samples,
            "eval_samples": config.eval_samples,
            "baseline_epochs": config.baseline_epochs,
            "bag_size": config.bag_size,
            "recovery_ratio": config.recovery_ratio,
            "lr": config.lr,
            "samples_per_launch": config.samples_per_launch,
            "tolerance": config.tolerance
        },
        "guardrails": {
            "no_public_calibration": true,
            "no_public_training_rows": true,
            "no_teacher": true,
            "no_external_inference": true,
            "no_fallback_returns": true,
            "does_not_claim_full_kernel_parity": true,
            "does_not_claim_training_lane_parity": true,
            "does_not_route_scheduler_to_metal": true
        },
        "timing": {
            "report_wall_ms": started.elapsed().as_secs_f64() * 1000.0
        },
        "external_inference_calls": 0,
        "teacher_used": false,
        "public_training_rows": 0
    })
}

pub fn train_standalone_symliquid_metal(
    config: StandaloneTrainConfig,
    samples_per_launch: usize,
) -> Result<StandaloneTrainReport> {
    let train_start = Instant::now();
    let mut timing_breakdown_ms = BTreeMap::new();
    let setup_start = Instant::now();
    let train_suite = generate_cgs_hard_suite(config.train_seed, config.cases_per_task);
    let eval_suite = generate_cgs_hard_suite(config.eval_seed, config.cases_per_task);
    let labels = standalone_output_vocab();
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.train_seed ^ 0xA53A_5EED);
    let mut readout = LinearReadout::new(config.hv_dim, labels.len(), &mut rng);
    let train_examples =
        prepare_training_examples(&train_suite, &labels, config.hv_dim, standalone_features);
    let (features, targets) = training_batch_tensor(&train_examples, config.hv_dim)?;
    timing_breakdown_ms.insert(
        "setup_and_feature_extraction".to_string(),
        setup_start.elapsed().as_millis(),
    );

    let metal_train_start = Instant::now();
    let trace = train_readout_sgd_metal(
        &features,
        &targets,
        &mut readout,
        config.epochs,
        config.lr,
        samples_per_launch,
    )?;
    timing_breakdown_ms.insert(
        "metal_readout_train".to_string(),
        metal_train_start.elapsed().as_millis(),
    );
    let train_runtime_ms = train_start.elapsed().as_millis();
    let trained_examples = train_examples.len() * config.epochs;
    let train_examples_per_second =
        trained_examples as f32 / (train_runtime_ms.max(1) as f32 / 1000.0);

    let eval_start = Instant::now();
    let eval = evaluate_standalone_model(
        &eval_suite,
        "symliquid-standalone-metal-readout",
        config.hv_dim,
        &labels,
        &readout,
        config.symbolic_fallback,
    )?;
    timing_breakdown_ms.insert("eval".to_string(), eval_start.elapsed().as_millis());
    timing_breakdown_ms.insert("total".to_string(), train_start.elapsed().as_millis());
    if let Some(path) = &config.artifact_path {
        write_readout_artifact(
            path,
            &readout,
            &labels,
            config.hv_dim,
            "structured_cgs_vsa_metal_readout",
        )?;
    }

    let mut runtime_profile = BTreeMap::new();
    runtime_profile.insert("backend".to_string(), "apple_metal".to_string());
    runtime_profile.insert("native_rust_owned".to_string(), "true".to_string());
    runtime_profile.insert("python_mlx_bridge_used".to_string(), "false".to_string());
    runtime_profile.insert("cuda_fallback".to_string(), "false".to_string());
    runtime_profile.insert(
        "production_scheduler_routing_enabled".to_string(),
        "false".to_string(),
    );

    Ok(StandaloneTrainReport {
        model_id: "symliquid-standalone-metal-readout".to_string(),
        feature_set: "structured_cgs_vsa_metal_readout".to_string(),
        train_seed: config.train_seed,
        eval_seed: config.eval_seed,
        cases_per_task: config.cases_per_task,
        epochs: config.epochs,
        batch_size: 1,
        hv_dim: config.hv_dim,
        labels: labels.len(),
        symbolic_fallback: config.symbolic_fallback,
        train_runtime_ms,
        train_examples_per_second,
        train_loss: trace.loss,
        train_accuracy: trace.accuracy,
        readout_routing: None,
        state_training: None,
        runtime_profile,
        timing_breakdown_ms,
        kernel_launches: config.epochs * features.rows.div_ceil(samples_per_launch.max(1)),
        cuda_fallback: false,
        eval,
    })
}

fn train_readout_sgd_cpu_reference(
    features: &symliquid_core::tensor::Tensor,
    targets: &[usize],
    readout: &mut symliquid_core::train::LinearReadout,
    epochs: usize,
    lr: f32,
) -> symliquid_core::error::Result<symliquid_core::train::TrainingTrace> {
    for _ in 0..epochs {
        for (row, target) in targets.iter().copied().enumerate().take(features.rows) {
            let sample = symliquid_core::tensor::Tensor::from_row(features.row(row).to_vec());
            readout.train_step(&sample, target, lr)?;
        }
    }
    readout.evaluate_batch(features, targets)
}

fn train_readout_bag_sgd_cpu_reference(
    features: &symliquid_core::tensor::Tensor,
    target_bags: &[usize],
    targets_per_sample: usize,
    readout: &mut symliquid_core::train::LinearReadout,
    epochs: usize,
    lr: f32,
) -> symliquid_core::error::Result<symliquid_core::train::TrainingTrace> {
    for _ in 0..epochs {
        for row in 0..features.rows {
            let sample = symliquid_core::tensor::Tensor::from_row(features.row(row).to_vec());
            let start = row.saturating_mul(targets_per_sample);
            let end = start.saturating_add(targets_per_sample);
            let bag = target_bags.get(start..end).ok_or_else(|| {
                symliquid_core::error::SymError::Shape(format!(
                    "target bag row {row} expected range {start}..{end}, got {} values",
                    target_bags.len()
                ))
            })?;
            readout.train_batch_target_bags(&sample, bag, targets_per_sample, lr)?;
        }
    }
    readout.train_batch_target_bags(features, target_bags, targets_per_sample, 0.0)
}

fn synthetic_token_superposition_dataset(
    config: &TokenSuperpositionMetalProofConfig,
) -> symliquid_core::token_superposition::TokenSuperpositionDataset {
    let vocab = (0..config.vocab_size)
        .map(|idx| format!("private_token_{idx:03}"))
        .collect::<Vec<_>>();
    let eval_len = config
        .eval_samples
        .saturating_add(config.bag_size)
        .saturating_add(8)
        .max(24);
    let train_tokens = deterministic_token_stream(
        config
            .train_tokens
            .max((2 * config.bag_size).saturating_add(8)),
        config.vocab_size,
        0x5453_5402,
    );
    let language_eval_tokens = deterministic_token_stream(eval_len, config.vocab_size, 0x5453_5403);
    let code_eval_tokens = deterministic_token_stream(eval_len, config.vocab_size, 0x5453_5404);
    let summary = symliquid_core::token_superposition::TokenSuperpositionDatasetSummary {
        language_train_docs: 1,
        language_eval_docs: 1,
        code_train_docs: 1,
        code_eval_docs: 1,
        train_tokens: train_tokens.len(),
        language_eval_tokens: language_eval_tokens.len(),
        code_eval_tokens: code_eval_tokens.len(),
        vocab_size: vocab.len(),
        hv_dim: config.hv_dim,
        holdout_policy: "Deterministic private synthetic train/eval split for Metal readout parity only; no public calibration or benchmark rows.".to_string(),
    };
    symliquid_core::token_superposition::TokenSuperpositionDataset {
        vocab,
        train_tokens,
        language_eval_tokens,
        code_eval_tokens,
        summary,
    }
}

fn deterministic_token_stream(len: usize, vocab_size: usize, salt: usize) -> Vec<usize> {
    (0..len)
        .map(|idx| {
            let mixed = idx
                .saturating_mul(37)
                .saturating_add((idx % 11).saturating_mul(17))
                .saturating_add((idx / 7).saturating_mul(5))
                .saturating_add(salt);
            mixed % vocab_size.max(1)
        })
        .collect()
}

fn token_superposition_readout(
    config: &TokenSuperpositionMetalProofConfig,
) -> symliquid_core::train::LinearReadout {
    symliquid_core::train::LinearReadout {
        input_dim: config.hv_dim,
        output_dim: config.vocab_size,
        weights: deterministic_values(config.vocab_size.saturating_mul(config.hv_dim), 0.018),
        bias: deterministic_values(config.vocab_size, 0.011),
    }
}

fn eval_trace_from_logits(
    logits: &symliquid_core::tensor::Tensor,
    targets: &[usize],
) -> symliquid_core::error::Result<symliquid_core::train::TrainingTrace> {
    if targets.len() != logits.rows {
        return Err(symliquid_core::error::SymError::Shape(format!(
            "target batch expected {}, got {}",
            logits.rows,
            targets.len()
        )));
    }
    let mut loss = 0.0;
    let mut correct = 0usize;
    for (row, target) in targets.iter().copied().enumerate() {
        if target >= logits.cols {
            return Err(symliquid_core::error::SymError::InvalidArgument(format!(
                "target {target} outside output dim {}",
                logits.cols
            )));
        }
        let probs = symliquid_core::tensor::softmax(logits.row(row));
        loss += -probs[target].max(1.0e-8).ln();
        correct += usize::from(argmax(&probs) == target);
    }
    let scale = 1.0 / targets.len().max(1) as f32;
    Ok(symliquid_core::train::TrainingTrace {
        loss: loss * scale,
        accuracy: correct as f32 * scale,
        grad_norm: 0.0,
    })
}

fn deterministic_readout(config: &RolloutMetalProofConfig) -> symliquid_core::train::LinearReadout {
    symliquid_core::train::LinearReadout {
        input_dim: config.hv_dim,
        output_dim: config.output_dim,
        weights: deterministic_values(config.output_dim.saturating_mul(config.hv_dim), 0.018),
        bias: deterministic_values(config.output_dim, 0.011),
    }
}

fn rollout_label_names(output_dim: usize) -> Vec<String> {
    (0..output_dim)
        .map(|index| format!("rollout_label_{index}"))
        .collect()
}

fn deterministic_targets(rows: usize, output_dim: usize) -> Vec<usize> {
    deterministic_targets_with_offset(rows, output_dim, 0)
}

fn deterministic_targets_with_offset(rows: usize, output_dim: usize, offset: usize) -> Vec<usize> {
    (0..rows)
        .map(|row| {
            offset
                .saturating_add(row)
                .saturating_mul(3)
                .saturating_add(1)
                % output_dim.max(1)
        })
        .collect()
}

fn accuracy(logits: &symliquid_core::tensor::Tensor, targets: &[usize]) -> f32 {
    if logits.rows == 0 || targets.is_empty() {
        return 0.0;
    }
    let correct = (0..logits.rows.min(targets.len()))
        .filter(|row| argmax(logits.row(*row)) == targets[*row])
        .count();
    correct as f32 / logits.rows.min(targets.len()).max(1) as f32
}

fn prediction_agreement(
    left: &symliquid_core::tensor::Tensor,
    right: &symliquid_core::tensor::Tensor,
) -> f32 {
    let rows = left.rows.min(right.rows);
    if rows == 0 {
        return 0.0;
    }
    let agree = (0..rows)
        .filter(|row| argmax(left.row(*row)) == argmax(right.row(*row)))
        .count();
    agree as f32 / rows as f32
}

fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (idx, value) in values.iter().copied().enumerate() {
        if value > best_value {
            best_idx = idx;
            best_value = value;
        }
    }
    best_idx
}

fn rollout_state(
    config: &RolloutMetalProofConfig,
) -> symliquid_core::error::Result<symliquid_cuda::rollout_cuda::RolloutState> {
    Ok(symliquid_cuda::rollout_cuda::RolloutState {
        h: symliquid_core::tensor::Tensor::new(
            config.batch,
            config.hidden_dim,
            deterministic_values(config.batch.saturating_mul(config.hidden_dim), 0.03),
        )?,
        r: symliquid_core::tensor::Tensor::new(
            config.batch,
            config.reservoir_dim,
            deterministic_values(config.batch.saturating_mul(config.reservoir_dim), 0.02),
        )?,
        memory: symliquid_core::tensor::Tensor::zeros(config.batch, config.hv_dim),
    })
}

fn rollout_params(config: &RolloutMetalProofConfig) -> symliquid_cuda::rollout_cuda::RolloutParams {
    symliquid_cuda::rollout_cuda::RolloutParams {
        obs_to_h: deterministic_values(config.hidden_dim.saturating_mul(config.obs_dim), 0.04),
        h_recurrent: deterministic_values(
            config.hidden_dim.saturating_mul(config.hidden_dim),
            0.015,
        ),
        h_bias: deterministic_values(config.hidden_dim, 0.02),
        reservoir_input: deterministic_values(
            config.reservoir_dim.saturating_mul(config.hidden_dim),
            0.03,
        ),
        reservoir_recurrent: deterministic_values(
            config.reservoir_dim.saturating_mul(config.reservoir_dim),
            0.01,
        ),
        reservoir_bias: deterministic_values(config.reservoir_dim, 0.015),
        hv_proj: deterministic_values(config.hv_dim.saturating_mul(config.reservoir_dim), 0.025),
    }
}

fn deterministic_values(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| (((idx * 37 + 11) % 23) as f32 - 11.0) * scale)
        .collect()
}

fn examples_per_second(examples: usize, runtime_ms: u128) -> f32 {
    examples as f32 / (runtime_ms.max(1) as f32 / 1000.0)
}

fn normalize_dense(values: &mut [f32]) {
    let norm = values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1.0e-6);
    for value in values {
        *value /= norm;
    }
}

fn max_abs_delta(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn max_row_abs_delta(
    left: &symliquid_core::tensor::Tensor,
    right: &symliquid_core::tensor::Tensor,
) -> Vec<f32> {
    let rows = left.rows.min(right.rows);
    (0..rows)
        .map(|row| max_abs_delta(left.row(row), right.row(row)))
        .collect()
}
