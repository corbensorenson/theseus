#![recursion_limit = "512"]

use std::fs;
use std::path::Path;

mod code_lm_closure;
mod code_ranker;
mod code_token_generator;
mod sts_parallel_decoder;

use clap::{Parser, Subcommand};

use symliquid_core::ablations::run_ablation_suite;
#[cfg(feature = "cuda")]
use symliquid_core::benchmarks::StandaloneTrainReport;
use symliquid_core::benchmarks::{
    compare_reports, format_babylm_probe_train_report, format_comparison, format_seed_sweep_report,
    format_summary, generate_babylm_probe_suite, generate_cgs_hard_suite,
    read_report_or_train_eval, read_suite_json, run_local_baseline_suite, run_seed_sweep,
    run_symliquid_suite, score_responses, train_babylm_probe_scorer, train_standalone_symliquid,
    train_text_hash_baseline, write_babylm_probe_train_report, write_breakdown_csv,
    write_comparison, write_report, write_request_template, write_seed_sweep_report,
    write_standalone_train_report, write_suite_json, BabyLmProbeTrainConfig, LocalBaselineKind,
    RunMode, StandaloneTrainConfig,
};
use symliquid_core::config::AblationConfig;
use symliquid_core::eval::{format_report, format_table};
use symliquid_core::tasks::active_classification::{run as run_active, ActiveClassificationConfig};
use symliquid_core::tasks::active_gridworld::{run as run_gridworld, ActiveGridworldConfig};
use symliquid_core::tasks::delayed_recall::{run as run_delayed, DelayedRecallConfig};
use symliquid_core::tasks::role_filler::{run as run_role, RoleFillerConfig};
#[cfg(feature = "cuda")]
use symliquid_core::token_superposition::write_token_superposition_report;
use symliquid_core::token_superposition::{
    format_token_superposition_report, TokenSuperpositionConfig,
};

use code_lm_closure::{
    generate_code_lm_closure_fanout, train_code_lm_closure, CodeLmClosureConfig, CodeLmFanoutConfig,
};
use code_ranker::{train_code_ranker, CodeRankerConfig};
use code_token_generator::{train_code_token_generator, CodeTokenGeneratorConfig};
use sts_parallel_decoder::{train_sts_parallel_decoder, StsParallelDecoderConfig};

#[derive(Debug, Parser)]
#[command(name = "symliquid")]
#[command(about = "Run SymLiquid FEP-Net reference tasks and ablations.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    RoleFiller {
        #[arg(long, default_value_t = 200)]
        steps: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "full")]
        variant: String,
    },
    DelayedRecall {
        #[arg(long, default_value_t = 200)]
        steps: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "full")]
        variant: String,
    },
    ActiveClassification {
        #[arg(long, default_value_t = 100)]
        episodes: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "full")]
        variant: String,
    },
    Gridworld {
        #[arg(long, default_value_t = 100)]
        episodes: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "full")]
        variant: String,
    },
    Ablations {
        #[arg(long, default_value = "role_filler")]
        task: String,
        #[arg(long, default_value_t = 200)]
        steps: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    BenchmarkSnapshot {
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 20)]
        cases_per_task: usize,
        #[arg(long, default_value = "benchmarks/snapshots/cgs_hard_seed0.json")]
        out: String,
    },
    BenchmarkSymliquid {
        #[arg(long, default_value = "benchmarks/snapshots/cgs_hard_seed0.json")]
        suite: String,
        #[arg(long, default_value = "reports/symliquid_reference_report.json")]
        out: String,
        #[arg(long, default_value = "symliquid-reference")]
        model_id: String,
        #[arg(long, default_value_t = false)]
        hybrid: bool,
    },
    BenchmarkTemplate {
        #[arg(long, default_value = "benchmarks/snapshots/cgs_hard_seed0.json")]
        suite: String,
        #[arg(long, default_value = "benchmarks/requests/local_requests.jsonl")]
        out: String,
        #[arg(long, default_value = "local-model")]
        model_id: String,
        #[arg(long, default_value_t = false)]
        hybrid: bool,
    },
    BenchmarkScore {
        #[arg(long, default_value = "benchmarks/snapshots/cgs_hard_seed0.json")]
        suite: String,
        #[arg(long)]
        responses: String,
        #[arg(long, default_value = "reports/local_response_report.json")]
        out: String,
        #[arg(long, default_value = "local-model")]
        model_id: String,
        #[arg(long, default_value = "local_baseline")]
        mode: String,
    },
    BenchmarkCompare {
        #[arg(long)]
        baseline: String,
        #[arg(long)]
        candidate: String,
        #[arg(long, default_value = "reports/comparison_report.json")]
        out: String,
    },
    BenchmarkBreakdown {
        #[arg(long, default_value = "benchmarks/snapshots/cgs_hard_seed0.json")]
        suite: String,
        #[arg(long)]
        report: String,
        #[arg(long, default_value = "reports/breakdown.csv")]
        out: String,
        #[arg(long, default_value = "task")]
        group_by: String,
    },
    TrainStandalone {
        #[arg(long, default_value_t = 0)]
        train_seed: u64,
        #[arg(long, default_value_t = 10000)]
        eval_seed: u64,
        #[arg(long, default_value_t = 100)]
        cases_per_task: usize,
        #[arg(long, default_value_t = 20)]
        epochs: usize,
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        #[arg(long, default_value_t = 4096)]
        hv_dim: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long, default_value_t = false)]
        symbolic_fallback: bool,
        #[arg(long)]
        model_out: Option<String>,
        #[arg(long, default_value = "reports/symliquid_standalone_train_report.json")]
        out: String,
    },
    TrainStandaloneCuda {
        #[arg(long, default_value_t = 0)]
        train_seed: u64,
        #[arg(long, default_value_t = 10000)]
        eval_seed: u64,
        #[arg(long, default_value_t = 100)]
        cases_per_task: usize,
        #[arg(long, default_value_t = 20)]
        epochs: usize,
        #[arg(long, default_value_t = 64)]
        samples_per_launch: usize,
        #[arg(long, default_value_t = 4096)]
        hv_dim: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long, default_value_t = false)]
        symbolic_fallback: bool,
        #[arg(long)]
        model_out: Option<String>,
        #[arg(
            long,
            default_value = "reports/symliquid_standalone_cuda_train_report.json"
        )]
        out: String,
    },
    TrainStandaloneMlx {
        #[arg(long, default_value_t = 0)]
        train_seed: u64,
        #[arg(long, default_value_t = 10000)]
        eval_seed: u64,
        #[arg(long, default_value_t = 100)]
        cases_per_task: usize,
        #[arg(long, default_value_t = 20)]
        epochs: usize,
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        #[arg(long, default_value_t = 64)]
        samples_per_launch: usize,
        #[arg(long, default_value_t = 4096)]
        hv_dim: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long, default_value_t = false)]
        symbolic_fallback: bool,
        #[arg(long)]
        model_out: Option<String>,
        #[arg(long, default_value = "data/babylm_blimp_filtered_train.jsonl")]
        train_input: String,
        #[arg(long, default_value = "data/babylm_mutated_holdout_seed55.jsonl")]
        eval_input: String,
        #[arg(long, default_value = "smoke")]
        profile: String,
        #[arg(
            long,
            default_value = "reports/symliquid_standalone_mlx_train_report.json"
        )]
        out: String,
    },
    TrainStandaloneMetal {
        #[arg(long, default_value_t = 0)]
        train_seed: u64,
        #[arg(long, default_value_t = 10000)]
        eval_seed: u64,
        #[arg(long, default_value_t = 100)]
        cases_per_task: usize,
        #[arg(long, default_value_t = 20)]
        epochs: usize,
        #[arg(long, default_value_t = 64)]
        samples_per_launch: usize,
        #[arg(long, default_value_t = 4096)]
        hv_dim: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long, default_value_t = false)]
        symbolic_fallback: bool,
        #[arg(long)]
        model_out: Option<String>,
        #[arg(
            long,
            default_value = "reports/symliquid_standalone_metal_train_report.json"
        )]
        out: String,
    },
    TrainRolloutCuda {
        #[arg(long, default_value_t = 0)]
        train_seed: u64,
        #[arg(long, default_value_t = 10000)]
        eval_seed: u64,
        #[arg(long, default_value_t = 20)]
        cases_per_task: usize,
        #[arg(long, default_value_t = 5)]
        epochs: usize,
        #[arg(long, default_value_t = 1)]
        state_epochs: usize,
        #[arg(long, default_value_t = 0.2)]
        state_lr: f32,
        #[arg(long, default_value_t = 32)]
        samples_per_launch: usize,
        #[arg(long, default_value_t = 0)]
        probe_cases_per_task: usize,
        #[arg(long, default_value_t = 16)]
        rollout_batch: usize,
        #[arg(long, default_value_t = 64)]
        obs_dim: usize,
        #[arg(long, default_value_t = 96)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 128)]
        reservoir_dim: usize,
        #[arg(long, default_value_t = 2048)]
        hv_dim: usize,
        #[arg(long, default_value_t = 64)]
        seq_len: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long)]
        model_out: Option<String>,
        #[arg(
            long,
            default_value = "reports/symliquid_rollout_cuda_train_report.json"
        )]
        out: String,
    },
    TrainRolloutMlx {
        #[arg(long, default_value_t = 0)]
        train_seed: u64,
        #[arg(long, default_value_t = 10000)]
        eval_seed: u64,
        #[arg(long, default_value_t = 20)]
        cases_per_task: usize,
        #[arg(long, default_value_t = 5)]
        epochs: usize,
        #[arg(long, default_value_t = 1)]
        state_epochs: usize,
        #[arg(long, default_value_t = 0.2)]
        state_lr: f32,
        #[arg(long, default_value_t = 32)]
        samples_per_launch: usize,
        #[arg(long, default_value_t = 0)]
        probe_cases_per_task: usize,
        #[arg(long, default_value_t = 16)]
        rollout_batch: usize,
        #[arg(long, default_value_t = 64)]
        obs_dim: usize,
        #[arg(long, default_value_t = 96)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 128)]
        reservoir_dim: usize,
        #[arg(long, default_value_t = 2048)]
        hv_dim: usize,
        #[arg(long, default_value_t = 64)]
        seq_len: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long)]
        model_out: Option<String>,
        #[arg(long, default_value = "smoke")]
        profile: String,
        #[arg(
            long,
            default_value = "reports/symliquid_rollout_mlx_train_report.json"
        )]
        out: String,
    },
    TrainRolloutMetal {
        #[arg(long, default_value_t = 0)]
        train_seed: u64,
        #[arg(long, default_value_t = 10000)]
        eval_seed: u64,
        #[arg(long, default_value_t = 20)]
        cases_per_task: usize,
        #[arg(long, default_value_t = 5)]
        epochs: usize,
        #[arg(long, default_value_t = 1)]
        state_epochs: usize,
        #[arg(long, default_value_t = 0.2)]
        state_lr: f32,
        #[arg(long, default_value_t = 32)]
        samples_per_launch: usize,
        #[arg(long, default_value_t = 0)]
        probe_cases_per_task: usize,
        #[arg(long, default_value_t = 16)]
        rollout_batch: usize,
        #[arg(long, default_value_t = 64)]
        obs_dim: usize,
        #[arg(long, default_value_t = 96)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 128)]
        reservoir_dim: usize,
        #[arg(long, default_value_t = 2048)]
        hv_dim: usize,
        #[arg(long, default_value_t = 64)]
        seq_len: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long, default_value_t = 4)]
        output_dim: usize,
        #[arg(long, default_value_t = 0.0001)]
        tolerance: f32,
        #[arg(long)]
        model_out: Option<String>,
        #[arg(long, default_value = "smoke")]
        profile: String,
        #[arg(
            long,
            default_value = "reports/symliquid_rollout_metal_train_report.json"
        )]
        out: String,
    },
    RolloutMetalProof {
        #[arg(long, default_value_t = 2)]
        batch: usize,
        #[arg(long, default_value_t = 3)]
        steps: usize,
        #[arg(long, default_value_t = 3)]
        obs_dim: usize,
        #[arg(long, default_value_t = 4)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 5)]
        reservoir_dim: usize,
        #[arg(long, default_value_t = 7)]
        hv_dim: usize,
        #[arg(long, default_value_t = 0.0001)]
        tolerance: f32,
        #[arg(
            long,
            default_value = "reports/macos_metal_rollout_hot_loop_proof.json"
        )]
        out: String,
    },
    RolloutMetalFeatureProof {
        #[arg(long, default_value_t = 6)]
        cases: usize,
        #[arg(long, default_value_t = 2)]
        batch: usize,
        #[arg(long, default_value_t = 3)]
        steps: usize,
        #[arg(long, default_value_t = 3)]
        obs_dim: usize,
        #[arg(long, default_value_t = 4)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 5)]
        reservoir_dim: usize,
        #[arg(long, default_value_t = 7)]
        hv_dim: usize,
        #[arg(long, default_value_t = 0.0001)]
        tolerance: f32,
        #[arg(long, default_value = "reports/macos_metal_rollout_feature_proof.json")]
        out: String,
    },
    RolloutMetalReadoutProof {
        #[arg(long, default_value_t = 6)]
        cases: usize,
        #[arg(long, default_value_t = 2)]
        batch: usize,
        #[arg(long, default_value_t = 3)]
        steps: usize,
        #[arg(long, default_value_t = 3)]
        obs_dim: usize,
        #[arg(long, default_value_t = 4)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 5)]
        reservoir_dim: usize,
        #[arg(long, default_value_t = 7)]
        hv_dim: usize,
        #[arg(long, default_value_t = 4)]
        output_dim: usize,
        #[arg(long, default_value_t = 0.0001)]
        tolerance: f32,
        #[arg(long, default_value = "reports/macos_metal_rollout_readout_proof.json")]
        out: String,
    },
    RolloutMetalReadoutTrainingProof {
        #[arg(long, default_value_t = 6)]
        cases: usize,
        #[arg(long, default_value_t = 2)]
        batch: usize,
        #[arg(long, default_value_t = 3)]
        steps: usize,
        #[arg(long, default_value_t = 3)]
        obs_dim: usize,
        #[arg(long, default_value_t = 4)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 5)]
        reservoir_dim: usize,
        #[arg(long, default_value_t = 7)]
        hv_dim: usize,
        #[arg(long, default_value_t = 4)]
        output_dim: usize,
        #[arg(long, default_value_t = 2)]
        readout_epochs: usize,
        #[arg(long, default_value_t = 0.03)]
        readout_lr: f32,
        #[arg(long, default_value_t = 2)]
        samples_per_launch: usize,
        #[arg(long, default_value_t = 0.0001)]
        tolerance: f32,
        #[arg(
            long,
            default_value = "reports/macos_metal_rollout_readout_training_proof.json"
        )]
        out: String,
    },
    RolloutMetalTrainPathProof {
        #[arg(long, default_value_t = 6)]
        cases: usize,
        #[arg(long, default_value_t = 2)]
        batch: usize,
        #[arg(long, default_value_t = 3)]
        steps: usize,
        #[arg(long, default_value_t = 3)]
        obs_dim: usize,
        #[arg(long, default_value_t = 4)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 5)]
        reservoir_dim: usize,
        #[arg(long, default_value_t = 7)]
        hv_dim: usize,
        #[arg(long, default_value_t = 4)]
        output_dim: usize,
        #[arg(long, default_value_t = 2)]
        readout_epochs: usize,
        #[arg(long, default_value_t = 0.03)]
        readout_lr: f32,
        #[arg(long, default_value_t = 2)]
        samples_per_launch: usize,
        #[arg(long, default_value_t = 0.0001)]
        tolerance: f32,
        #[arg(long, default_value = "reports/macos_metal_train_path_proof.json")]
        out: String,
    },
    RolloutMetalStateTrainingProof {
        #[arg(long, default_value_t = 6)]
        cases: usize,
        #[arg(long, default_value_t = 2)]
        batch: usize,
        #[arg(long, default_value_t = 3)]
        steps: usize,
        #[arg(long, default_value_t = 3)]
        obs_dim: usize,
        #[arg(long, default_value_t = 4)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 5)]
        reservoir_dim: usize,
        #[arg(long, default_value_t = 7)]
        hv_dim: usize,
        #[arg(long, default_value_t = 4)]
        output_dim: usize,
        #[arg(long, default_value_t = 2)]
        state_epochs: usize,
        #[arg(long, default_value_t = 0.03)]
        state_lr: f32,
        #[arg(long, default_value_t = 0.0005)]
        tolerance: f32,
        #[arg(
            long,
            default_value = "reports/macos_metal_rollout_state_training_proof.json"
        )]
        out: String,
    },
    TokenSuperpositionMetalReadoutProof {
        #[arg(long, default_value_t = 16)]
        vocab_size: usize,
        #[arg(long, default_value_t = 32)]
        hv_dim: usize,
        #[arg(long, default_value_t = 192)]
        train_tokens: usize,
        #[arg(long, default_value_t = 24)]
        train_samples: usize,
        #[arg(long, default_value_t = 24)]
        eval_samples: usize,
        #[arg(long, default_value_t = 2)]
        baseline_epochs: usize,
        #[arg(long, default_value_t = 4)]
        bag_size: usize,
        #[arg(long, default_value_t = 0.5)]
        recovery_ratio: f32,
        #[arg(long, default_value_t = 0.03)]
        lr: f32,
        #[arg(long, default_value_t = 4)]
        samples_per_launch: usize,
        #[arg(long, default_value_t = 0.0005)]
        tolerance: f32,
        #[arg(
            long,
            default_value = "reports/macos_metal_token_superposition_readout_proof.json"
        )]
        out: String,
    },
    TrainRolloutCudaSweep {
        #[arg(long, default_value = "0,1,2")]
        train_seeds: String,
        #[arg(long, default_value_t = 10000)]
        eval_seed_base: u64,
        #[arg(long, default_value_t = 50)]
        cases_per_task: usize,
        #[arg(long, default_value_t = 5)]
        epochs: usize,
        #[arg(long, default_value = "0,2,6")]
        state_epochs: String,
        #[arg(long, default_value = "0.0,0.005,0.02")]
        state_lrs: String,
        #[arg(long, default_value_t = 32)]
        samples_per_launch: usize,
        #[arg(long, default_value_t = 0)]
        probe_cases_per_task: usize,
        #[arg(long, default_value_t = 200)]
        rollout_batch: usize,
        #[arg(long, default_value_t = 64)]
        obs_dim: usize,
        #[arg(long, default_value_t = 96)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 128)]
        reservoir_dim: usize,
        #[arg(long, default_value_t = 1024)]
        hv_dim: usize,
        #[arg(long, default_value_t = 64)]
        seq_len: usize,
        #[arg(long, default_value_t = 0.03)]
        lr: f32,
        #[arg(long, default_value = "reports/symliquid_rollout_cuda_sweep.json")]
        out: String,
    },
    TrainRolloutMlxSweep {
        #[arg(long, default_value = "0,1,2")]
        train_seeds: String,
        #[arg(long, default_value_t = 10000)]
        eval_seed_base: u64,
        #[arg(long, default_value_t = 50)]
        cases_per_task: usize,
        #[arg(long, default_value_t = 5)]
        epochs: usize,
        #[arg(long, default_value = "0,2,6")]
        state_epochs: String,
        #[arg(long, default_value = "0.0,0.005,0.02")]
        state_lrs: String,
        #[arg(long, default_value_t = 32)]
        samples_per_launch: usize,
        #[arg(long, default_value_t = 0)]
        probe_cases_per_task: usize,
        #[arg(long, default_value_t = 200)]
        rollout_batch: usize,
        #[arg(long, default_value_t = 64)]
        obs_dim: usize,
        #[arg(long, default_value_t = 96)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 128)]
        reservoir_dim: usize,
        #[arg(long, default_value_t = 1024)]
        hv_dim: usize,
        #[arg(long, default_value_t = 64)]
        seq_len: usize,
        #[arg(long, default_value_t = 0.03)]
        lr: f32,
        #[arg(long, default_value = "smoke")]
        profile: String,
        #[arg(long, default_value = "reports/symliquid_rollout_mlx_sweep.json")]
        out: String,
    },
    TrainRolloutMetalSweep {
        #[arg(long, default_value = "0,1,2")]
        train_seeds: String,
        #[arg(long, default_value_t = 10000)]
        eval_seed_base: u64,
        #[arg(long, default_value_t = 50)]
        cases_per_task: usize,
        #[arg(long, default_value_t = 5)]
        epochs: usize,
        #[arg(long, default_value = "0,2,6")]
        state_epochs: String,
        #[arg(long, default_value = "0.0,0.005,0.02")]
        state_lrs: String,
        #[arg(long, default_value_t = 32)]
        samples_per_launch: usize,
        #[arg(long, default_value_t = 0)]
        probe_cases_per_task: usize,
        #[arg(long, default_value_t = 200)]
        rollout_batch: usize,
        #[arg(long, default_value_t = 64)]
        obs_dim: usize,
        #[arg(long, default_value_t = 96)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 128)]
        reservoir_dim: usize,
        #[arg(long, default_value_t = 1024)]
        hv_dim: usize,
        #[arg(long, default_value_t = 64)]
        seq_len: usize,
        #[arg(long, default_value_t = 0.03)]
        lr: f32,
        #[arg(long, default_value_t = 4)]
        output_dim: usize,
        #[arg(long, default_value_t = 0.0001)]
        tolerance: f32,
        #[arg(long, default_value = "smoke")]
        profile: String,
        #[arg(long, default_value = "reports/macos_metal_rollout_sweep_artifacts")]
        artifact_dir: String,
        #[arg(long, default_value = "reports/symliquid_rollout_metal_sweep.json")]
        out: String,
    },
    TrainTokenSuperpositionCuda {
        #[arg(long, default_value = "data/babylm_blimp_filtered_train.jsonl")]
        input: String,
        #[arg(long, default_value_t = false)]
        include_project_code: bool,
        #[arg(long, default_value = "scripts,crates")]
        project_code_roots: String,
        #[arg(long, default_value_t = 20260514)]
        train_seed: u64,
        #[arg(long, default_value_t = 8000)]
        max_language_rows: usize,
        #[arg(long, default_value_t = 160)]
        max_code_files: usize,
        #[arg(long, default_value_t = 12000)]
        max_chars_per_doc: usize,
        #[arg(long, default_value_t = 256)]
        max_vocab: usize,
        #[arg(long, default_value_t = 4096)]
        hv_dim: usize,
        #[arg(long, default_value_t = 32768)]
        train_samples: usize,
        #[arg(long, default_value_t = 4096)]
        eval_samples: usize,
        #[arg(long, default_value_t = 6)]
        baseline_epochs: usize,
        #[arg(long, default_value = "4,8")]
        bag_sizes: String,
        #[arg(long, default_value = "0.2,0.4")]
        recovery_ratios: String,
        #[arg(long, default_value_t = 0.03)]
        lr: f32,
        #[arg(long, default_value_t = 512)]
        samples_per_launch: usize,
        #[arg(long, default_value_t = 0.002)]
        gate_tolerance: f32,
        #[arg(long, default_value_t = 1.2)]
        min_nominal_speedup: f32,
        #[arg(long, default_value_t = 1.0)]
        min_train_speedup: f32,
        #[arg(long)]
        model_out: Option<String>,
        #[arg(long, default_value = "reports/token_superposition_training.json")]
        out: String,
    },
    TrainTokenSuperpositionMlx {
        #[arg(long, default_value = "data/babylm_blimp_filtered_train.jsonl")]
        input: String,
        #[arg(long, default_value_t = false)]
        include_project_code: bool,
        #[arg(long, default_value = "scripts,crates")]
        project_code_roots: String,
        #[arg(long, default_value_t = 20260514)]
        train_seed: u64,
        #[arg(long, default_value_t = 8000)]
        max_language_rows: usize,
        #[arg(long, default_value_t = 160)]
        max_code_files: usize,
        #[arg(long, default_value_t = 12000)]
        max_chars_per_doc: usize,
        #[arg(long, default_value_t = 256)]
        max_vocab: usize,
        #[arg(long, default_value_t = 4096)]
        hv_dim: usize,
        #[arg(long, default_value_t = 32768)]
        train_samples: usize,
        #[arg(long, default_value_t = 4096)]
        eval_samples: usize,
        #[arg(long, default_value_t = 6)]
        baseline_epochs: usize,
        #[arg(long, default_value = "4,8")]
        bag_sizes: String,
        #[arg(long, default_value = "0.2,0.4")]
        recovery_ratios: String,
        #[arg(long, default_value_t = 0.03)]
        lr: f32,
        #[arg(long, default_value_t = 512)]
        samples_per_launch: usize,
        #[arg(long, default_value_t = 0.002)]
        gate_tolerance: f32,
        #[arg(long, default_value_t = 1.2)]
        min_nominal_speedup: f32,
        #[arg(long, default_value_t = 1.0)]
        min_train_speedup: f32,
        #[arg(long)]
        model_out: Option<String>,
        #[arg(long, default_value = "reports/token_superposition_mlx_training.json")]
        out: String,
    },
    TrainTokenSuperpositionMetal {
        #[arg(long, default_value = "data/babylm_blimp_filtered_train.jsonl")]
        input: String,
        #[arg(long, default_value_t = false)]
        include_project_code: bool,
        #[arg(long, default_value = "scripts,crates")]
        project_code_roots: String,
        #[arg(long, default_value_t = 20260514)]
        train_seed: u64,
        #[arg(long, default_value_t = 8000)]
        max_language_rows: usize,
        #[arg(long, default_value_t = 160)]
        max_code_files: usize,
        #[arg(long, default_value_t = 12000)]
        max_chars_per_doc: usize,
        #[arg(long, default_value_t = 256)]
        max_vocab: usize,
        #[arg(long, default_value_t = 4096)]
        hv_dim: usize,
        #[arg(long, default_value_t = 32768)]
        train_samples: usize,
        #[arg(long, default_value_t = 4096)]
        eval_samples: usize,
        #[arg(long, default_value_t = 6)]
        baseline_epochs: usize,
        #[arg(long, default_value = "4,8")]
        bag_sizes: String,
        #[arg(long, default_value = "0.2,0.4")]
        recovery_ratios: String,
        #[arg(long, default_value_t = 0.03)]
        lr: f32,
        #[arg(long, default_value_t = 512)]
        samples_per_launch: usize,
        #[arg(long, default_value_t = 0.002)]
        gate_tolerance: f32,
        #[arg(long, default_value_t = 1.2)]
        min_nominal_speedup: f32,
        #[arg(long, default_value_t = 1.0)]
        min_train_speedup: f32,
        #[arg(long)]
        model_out: Option<String>,
        #[arg(
            long,
            default_value = "reports/token_superposition_metal_training.json"
        )]
        out: String,
    },
    TrainCodeRanker {
        #[arg(long, default_value = "reports/student_code_candidates.jsonl")]
        candidate_manifest: String,
        #[arg(long, default_value = "reports/real_code_benchmark_traces.jsonl")]
        trace_in: String,
        #[arg(long, default_value_t = 14)]
        seed: u64,
        #[arg(long, default_value_t = 0.34)]
        holdout_ratio: f32,
        #[arg(long, default_value_t = 10)]
        epochs: usize,
        #[arg(long, default_value_t = 512)]
        hv_dim: usize,
        #[arg(long, default_value_t = 0.15)]
        lr: f32,
        #[arg(long, default_value_t = false)]
        use_cuda_readout: bool,
        #[arg(long, default_value = "reports/student_neural_code_checkpoint.json")]
        model_out: String,
        #[arg(long, default_value = "reports/student_learning_code_candidates.jsonl")]
        candidate_out: String,
        #[arg(
            long,
            default_value = "reports/student_learning_training_examples.jsonl"
        )]
        training_examples_out: String,
        #[arg(
            long,
            default_value = "reports/transfer_artifacts/code/student_neural_learning_closure_transfer_artifact.json"
        )]
        transfer_artifact_out: String,
        #[arg(long, default_value = "reports/code_transfer_artifacts.json")]
        code_transfer_artifacts: String,
        #[arg(long, default_value = "reports/student_learning_closure.json")]
        out: String,
    },
    TrainCodeTokenGenerator {
        #[arg(long, default_value = "reports/student_token_code_tasks.jsonl")]
        task_manifest: String,
        #[arg(
            long,
            default_value = "data/training_sources/old_project_registry_training_sources.json"
        )]
        training_sources: String,
        #[arg(long, default_value = "scripts,crates")]
        project_code_roots: String,
        #[arg(long, default_value_t = 14)]
        seed: u64,
        #[arg(long, default_value_t = 1200)]
        max_training_rows_per_source: usize,
        #[arg(long, default_value_t = 160)]
        max_project_files: usize,
        #[arg(long, default_value_t = 8)]
        max_candidates_per_task: usize,
        #[arg(long, default_value = "reports/student_token_code_checkpoint.json")]
        checkpoint_out: String,
        #[arg(long, default_value = "reports/student_code_candidates.jsonl")]
        out: String,
        #[arg(long, default_value = "reports/student_token_code_generator.json")]
        report_out: String,
    },
    TrainCodeLmClosure {
        #[arg(
            long,
            default_value = "data/private_code_curriculum/code_lm_closure_seed14.jsonl"
        )]
        private_curriculum: String,
        #[arg(long, default_value = "reports/code_lm_public_tasks.jsonl")]
        public_task_manifest: String,
        #[arg(long, default_value_t = 14)]
        seed: u64,
        #[arg(long, default_value_t = 512)]
        hv_dim: usize,
        #[arg(long, default_value_t = 320)]
        max_vocab: usize,
        #[arg(long, default_value_t = 5)]
        epochs: usize,
        #[arg(long, default_value_t = 0.08)]
        lr: f32,
        #[arg(long, default_value_t = 8)]
        candidates_per_task: usize,
        #[arg(long, default_value_t = 0)]
        max_work_steps: usize,
        #[arg(long, default_value_t = false)]
        use_cuda_readout: bool,
        #[arg(long, default_value_t = 4096)]
        readout_eval_limit: usize,
        #[arg(long, default_value_t = 512)]
        aux_decoder_train_limit: usize,
        #[arg(long, default_value_t = false)]
        checkpoint_only: bool,
        #[arg(long, default_value = "reports/student_code_lm_checkpoint.json")]
        checkpoint_out: String,
        #[arg(long, default_value = "")]
        checkpoint_in: String,
        #[arg(long, default_value = "reports/code_lm_private_candidates.jsonl")]
        private_candidate_out: String,
        #[arg(long, default_value = "reports/student_code_candidates.jsonl")]
        public_candidate_out: String,
        #[arg(long, default_value = "reports/code_lm_closure_rust.json")]
        report_out: String,
        #[arg(long, default_value = "")]
        sts_streams: String,
    },
    GenerateCodeLmClosureFanout {
        #[arg(
            long,
            default_value = "data/private_code_curriculum/code_lm_closure_seed14.jsonl"
        )]
        private_curriculum: String,
        #[arg(long, default_value = "reports/code_lm_public_tasks.jsonl")]
        public_task_manifest: String,
        #[arg(long, default_value = "reports/student_code_lm_checkpoint.json")]
        checkpoint_in: String,
        #[arg(long, default_value_t = 14)]
        seed: u64,
        #[arg(long, default_value_t = 8)]
        candidates_per_task: usize,
        #[arg(long, default_value = "reports/code_lm_private_candidates.jsonl")]
        private_candidate_out: String,
        #[arg(long, default_value = "reports/student_code_candidates.jsonl")]
        public_candidate_out: String,
        #[arg(long, default_value = "reports/code_lm_closure_rust_fanout.json")]
        report_out: String,
        #[arg(long, default_value = "")]
        sts_streams: String,
        #[arg(long, default_value = "")]
        transformer_hybrid_candidate_manifest: String,
        #[arg(long, default_value_t = 0)]
        private_eval_limit: usize,
        #[arg(long, default_value_t = 0)]
        public_task_limit: usize,
    },
    TrainStsParallelDecoder {
        #[arg(
            long,
            default_value = "data/sts_learning/sts_code_streams_seed14.jsonl"
        )]
        input: String,
        #[arg(long, default_value_t = 14)]
        seed: u64,
        #[arg(long, default_value_t = 384)]
        hv_dim: usize,
        #[arg(long, default_value_t = 384)]
        max_vocab: usize,
        #[arg(long, default_value_t = 3)]
        epochs: usize,
        #[arg(long, default_value_t = 0.06)]
        lr: f32,
        #[arg(long, default_value_t = 18)]
        max_generate_steps: usize,
        #[arg(long, default_value_t = 240)]
        max_train_rows: usize,
        #[arg(long, default_value_t = 80)]
        max_eval_rows: usize,
        #[arg(long, default_value_t = 128)]
        max_generate_rows: usize,
        #[arg(long, default_value = "reports/sts_parallel_decoder_checkpoint.json")]
        checkpoint_out: String,
        #[arg(long, default_value = "reports/sts_parallel_decoder_generations.jsonl")]
        generation_out: String,
        #[arg(long, default_value = "reports/sts_native_parallel_probe.json")]
        report_out: String,
    },
    TrainBaseline {
        #[arg(long, default_value_t = 0)]
        train_seed: u64,
        #[arg(long, default_value_t = 10000)]
        eval_seed: u64,
        #[arg(long, default_value_t = 100)]
        cases_per_task: usize,
        #[arg(long, default_value_t = 20)]
        epochs: usize,
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        #[arg(long, default_value_t = 4096)]
        hv_dim: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long)]
        model_out: Option<String>,
        #[arg(long, default_value = "reports/local_text_hash_train_report.json")]
        out: String,
    },
    SeedSweep {
        #[arg(long, default_value = "0,1,2")]
        train_seeds: String,
        #[arg(long, default_value_t = 10000)]
        eval_seed_base: u64,
        #[arg(long, default_value_t = 20)]
        cases_per_task: usize,
        #[arg(long, default_value_t = 10)]
        epochs: usize,
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        #[arg(long, default_value_t = 2048)]
        hv_dim: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long, default_value_t = false)]
        symbolic_fallback: bool,
        #[arg(long, default_value = "reports/symliquid_seed_sweep.json")]
        out: String,
    },
    BenchmarkBaseline {
        #[arg(long, default_value = "benchmarks/snapshots/cgs_hard_seed0.json")]
        suite: String,
        #[arg(long, default_value = "bag_of_words")]
        baseline: String,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 2048)]
        hv_dim: usize,
        #[arg(long, default_value_t = 10)]
        epochs: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long, default_value = "reports/local_baseline_report.json")]
        out: String,
    },
    BabylmProbe {
        #[arg(
            long,
            default_value = "C:\\Users\\corbe\\Documents\\babylm-candidate\\data\\samples\\strict_small_50k_words.txt"
        )]
        input: String,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 100)]
        limit: usize,
        #[arg(long, default_value = "benchmarks/snapshots/babylm_local_probe.json")]
        out_suite: String,
        #[arg(long, default_value = "reports/babylm_local_probe_report.json")]
        out_report: String,
    },
    TrainBabylmProbe {
        #[arg(
            long,
            default_value = "C:\\Users\\corbe\\Documents\\babylm-candidate\\data\\samples\\strict_small_50k_words.txt"
        )]
        input: String,
        #[arg(long)]
        eval_input: Option<String>,
        #[arg(long, default_value_t = 0)]
        train_seed: u64,
        #[arg(long, default_value_t = 10000)]
        eval_seed: u64,
        #[arg(long, default_value_t = 1000)]
        train_limit: usize,
        #[arg(long, default_value_t = 300)]
        eval_limit: usize,
        #[arg(long, default_value_t = 1000)]
        steps: usize,
        #[arg(long, default_value_t = 8192)]
        hv_dim: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long, default_value_t = false)]
        stateful: bool,
        #[arg(long, default_value_t = false)]
        pairwise_contrast: bool,
        #[arg(long, default_value_t = false)]
        balance_rules: bool,
        #[arg(long, default_value_t = 0.0)]
        prior_weight: f32,
        #[arg(long, default_value = "reports/babylm_probe_train_report.json")]
        out: String,
    },
}

#[cfg(feature = "cuda")]
#[derive(Debug, serde::Serialize)]
struct RolloutCudaSweepReport {
    train_seeds: Vec<u64>,
    eval_seed_base: u64,
    cases_per_task: usize,
    epochs: usize,
    state_epoch_grid: Vec<usize>,
    state_lr_grid: Vec<f32>,
    samples_per_launch: usize,
    probe_cases_per_task: usize,
    rollout_batch: usize,
    obs_dim: usize,
    hidden_dim: usize,
    reservoir_dim: usize,
    hv_dim: usize,
    seq_len: usize,
    lr: f32,
    runs: Vec<StandaloneTrainReport>,
    best_index: usize,
    best_accuracy: f32,
    best_residual: f32,
    mean_accuracy: f32,
    std_accuracy: f32,
    mean_residual: f32,
    std_residual: f32,
    accepted_state_candidates: usize,
}

#[cfg(feature = "cuda")]
struct RolloutCudaSweepBuild {
    train_seeds: Vec<u64>,
    eval_seed_base: u64,
    cases_per_task: usize,
    epochs: usize,
    state_epoch_grid: Vec<usize>,
    state_lr_grid: Vec<f32>,
    samples_per_launch: usize,
    probe_cases_per_task: usize,
    rollout_batch: usize,
    obs_dim: usize,
    hidden_dim: usize,
    reservoir_dim: usize,
    hv_dim: usize,
    seq_len: usize,
    lr: f32,
    runs: Vec<StandaloneTrainReport>,
}

fn large_stack_size_bytes() -> usize {
    std::env::var("RUST_MIN_STACK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 8 * 1024 * 1024)
        .unwrap_or(256 * 1024 * 1024)
}

fn mlx_bridge_args(command: &str) -> Vec<String> {
    vec![
        "scripts/macos_mlx_training.py".to_string(),
        command.to_string(),
    ]
}

fn push_cli_arg<T: ToString>(args: &mut Vec<String>, name: &str, value: T) {
    args.push(name.to_string());
    args.push(value.to_string());
}

fn write_json_value(
    path: &str,
    value: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn annotate_train_standalone_metal_report(report: &mut serde_json::Value, args: serde_json::Value) {
    let symbolic_fallback = report
        .get("symbolic_fallback")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let kernel_launches = report
        .get("kernel_launches")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let train_accuracy = report
        .get("train_accuracy")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let eval_accuracy = report
        .pointer("/eval/summary/accuracy")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let train_loss = report
        .get("train_loss")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let train_examples_per_second = report
        .get("train_examples_per_second")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let claimed_work_units = report
        .get("kernel_launches")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let artifact_write = standalone_metal_artifact_write_from_args(&args);
    if let Some(object) = report.as_object_mut() {
        object.insert("ok".to_string(), serde_json::json!(true));
        object.insert("state".to_string(), serde_json::json!("GREEN"));
        object.insert(
            "policy".to_string(),
            serde_json::json!("project_theseus_macos_metal_train_standalone_cli_report_v0"),
        );
        object.insert(
            "command".to_string(),
            serde_json::json!("train-standalone-metal"),
        );
        object.insert(
            "parity_for".to_string(),
            serde_json::json!("train-standalone-cuda"),
        );
        object.insert(
            "implementation".to_string(),
            serde_json::json!("rust_metal_structured_cgs_vsa_readout_cli"),
        );
        object.insert("backend".to_string(), serde_json::json!("apple_metal"));
        object.insert("cuda_fallback".to_string(), serde_json::json!(false));
        object.insert("args".to_string(), args);
        object.insert(
            "metrics".to_string(),
            serde_json::json!({
                "train_accuracy": train_accuracy,
                "eval_accuracy": eval_accuracy,
                "train_loss": train_loss,
                "train_examples_per_second": train_examples_per_second,
                "kernel_launches": kernel_launches
            }),
        );
        object.insert(
            "work_receipt".to_string(),
            serde_json::json!({
                "accepted": true,
                "backend": "apple_metal",
                "task_kind": "train_standalone_metal_cli",
                "worker_kind": "rust_metal_structured_cgs_vsa_readout",
                "claimed_work_units": claimed_work_units
            }),
        );
        object.insert("artifact_write".to_string(), artifact_write);
        object.insert(
            "promotion_decision".to_string(),
            serde_json::json!({
                "promote_to_training_lane": false,
                "status": "not_promoted_keep_as_native_contract_evidence",
                "reason": "Mac Rust/Metal standalone report contract exists, but full native parity, scheduler routing, public transfer, and promotion gates remain locked.",
                "artifact": ""
            }),
        );
        object.insert(
            "model_promotion_allowed".to_string(),
            serde_json::json!(false),
        );
        object.insert(
            "train_standalone_parity_claim_allowed".to_string(),
            serde_json::json!(false),
        );
        object.insert(
            "full_cli_parity_claim_allowed".to_string(),
            serde_json::json!(false),
        );
        object.insert("teacher_used".to_string(), serde_json::json!(false));
        object.insert("public_training_rows".to_string(), serde_json::json!(0));
        object.insert("external_inference_calls".to_string(), serde_json::json!(0));
        object.insert(
            "report_contract".to_string(),
            serde_json::json!({
                "matches_train_standalone_cli_surface": true,
                "mirrors_command": "train-standalone-mlx",
                "scheduler_routing_enabled": false,
                "scheduler_routing_blocker": "Do not route production scheduler work to Metal until full parity, route policy, and rollback guardrails are proven.",
                "python_mlx_bridge_used": false,
                "native_readout_subpath": "readout_sgd_samples_kernel + linear_readout_logits_kernel"
            }),
        );
        object.insert(
            "guardrails".to_string(),
            serde_json::json!({
                "no_public_calibration": true,
                "no_public_training_rows": true,
                "no_teacher": true,
                "no_external_inference": true,
                "no_fallback_returns": !symbolic_fallback,
                "does_not_claim_full_kernel_parity": true,
                "does_not_claim_training_lane_parity": true,
                "does_not_route_scheduler_to_metal": true,
                "promotion_locked_by_macos_contract": true
            }),
        );
    }
}

fn annotate_train_rollout_metal_report(report: &mut serde_json::Value, args: serde_json::Value) {
    let ok = report
        .get("ok")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let kernel_launches = report
        .get("kernel_launches")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let metric_train_accuracy = get_json_f64(report, &["train_metrics", "metal_accuracy"]);
    let metric_eval_accuracy = get_json_f64(report, &["eval_metrics", "metal_accuracy"]);
    let metric_train_loss = get_json_f64(report, &["train_metrics", "metal_loss"]);
    let metric_eval_loss = get_json_f64(report, &["eval_metrics", "metal_loss"]);
    let metric_train_examples_per_second =
        get_json_f64(report, &["timing", "metal_train_examples_per_second"]);
    let claimed_work_units = get_json_u64(report, &["kernel_launches"]).unwrap_or(0);
    let artifact_write = report
        .get("artifact_write")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    if let Some(object) = report.as_object_mut() {
        object.insert(
            "policy".to_string(),
            serde_json::json!("project_theseus_macos_metal_train_rollout_cli_report_v0"),
        );
        object.insert(
            "command".to_string(),
            serde_json::json!("train-rollout-metal"),
        );
        object.insert(
            "parity_for".to_string(),
            serde_json::json!("train-rollout-cuda"),
        );
        object.insert(
            "implementation".to_string(),
            serde_json::json!("rust_metal_rollout_readout_training_cli"),
        );
        object.insert("backend".to_string(), serde_json::json!("apple_metal"));
        object.insert("cuda_fallback".to_string(), serde_json::json!(false));
        object.insert("args".to_string(), args);
        object.insert(
            "metrics".to_string(),
            serde_json::json!({
                "train_accuracy": metric_train_accuracy,
                "eval_accuracy": metric_eval_accuracy,
                "train_loss": metric_train_loss,
                "eval_loss": metric_eval_loss,
                "train_examples_per_second": metric_train_examples_per_second,
                "kernel_launches": kernel_launches
            }),
        );
        object.insert(
            "work_receipt".to_string(),
            serde_json::json!({
                "accepted": ok,
                "backend": "apple_metal",
                "task_kind": "train_rollout_metal_cli",
                "worker_kind": "rust_metal_train_rollout",
                "claimed_work_units": claimed_work_units
            }),
        );
        object.insert(
            "promotion_decision".to_string(),
            serde_json::json!({
                "promote_to_training_lane": false,
                "status": "not_promoted_keep_as_native_contract_evidence",
                "reason": "Mac Rust/Metal report contract exists, but full scheduler routing and checkpoint/artifact equivalence are not proven."
            }),
        );
        object.insert("artifact_write".to_string(), artifact_write);
        object.insert(
            "report_contract".to_string(),
            serde_json::json!({
                "matches_train_rollout_cli_surface": true,
                "mirrors_command": "train-rollout-mlx",
                "scheduler_routing_enabled": false,
                "scheduler_routing_blocker": "Do not route production scheduler work to Metal until explicit route policy and rollback guardrails are added.",
                "python_mlx_bridge_used": false
            }),
        );
    }
}

fn annotate_train_token_superposition_metal_report(
    report: &mut serde_json::Value,
    args: serde_json::Value,
) {
    let raw_promotion = report
        .get("promotion_decision")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let baseline_kernel_launches =
        get_json_u64(report, &["baseline", "kernel_launches"]).unwrap_or(0);
    let variant_kernel_launches = report
        .get("variants")
        .and_then(|value| value.as_array())
        .map(|rows| {
            rows.iter()
                .filter_map(|row| row.get("kernel_launches").and_then(|value| value.as_u64()))
                .sum::<u64>()
        })
        .unwrap_or(0);
    let claimed_work_units = baseline_kernel_launches.saturating_add(variant_kernel_launches);
    let best_variant = report
        .get("best_variant")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let raw_policy = report
        .get("policy")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let artifact_write = token_superposition_artifact_write_from_args(&args);
    let baseline_combined_loss = get_json_f64(report, &["baseline", "eval", "combined_ar_loss"]);
    let baseline_code_loss = get_json_f64(report, &["baseline", "eval", "code_ar_loss"]);
    let baseline_train_examples_per_second =
        get_json_f64(report, &["baseline", "train_examples_per_second"]);
    let best_variant_id = best_variant
        .get("id")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let best_variant_combined_loss = best_variant
        .get("eval")
        .and_then(|eval| eval.get("combined_ar_loss"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let best_variant_code_loss = best_variant
        .get("eval")
        .and_then(|eval| eval.get("code_ar_loss"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let best_variant_train_examples_per_second = best_variant
        .get("train_examples_per_second")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    if let Some(object) = report.as_object_mut() {
        object.insert("ok".to_string(), serde_json::json!(true));
        object.insert(
            "policy".to_string(),
            serde_json::json!("project_theseus_token_superposition_metal_report_v1"),
        );
        object.insert("raw_core_policy".to_string(), raw_policy);
        object.insert(
            "command".to_string(),
            serde_json::json!("train-token-superposition-metal"),
        );
        object.insert(
            "parity_for".to_string(),
            serde_json::json!("train-token-superposition-cuda"),
        );
        object.insert(
            "implementation".to_string(),
            serde_json::json!("rust_metal_token_superposition_readout_cli"),
        );
        object.insert("backend".to_string(), serde_json::json!("apple_metal"));
        object.insert("cuda_fallback".to_string(), serde_json::json!(false));
        object.insert("args".to_string(), args);
        object.insert(
            "metrics".to_string(),
            serde_json::json!({
                "baseline_combined_loss": baseline_combined_loss,
                "baseline_code_loss": baseline_code_loss,
                "baseline_train_examples_per_second": baseline_train_examples_per_second,
                "best_variant_id": best_variant_id,
                "best_variant_combined_loss": best_variant_combined_loss,
                "best_variant_code_loss": best_variant_code_loss,
                "best_variant_train_examples_per_second": best_variant_train_examples_per_second,
                "kernel_launches": claimed_work_units
            }),
        );
        object.insert(
            "work_receipt".to_string(),
            serde_json::json!({
                "accepted": true,
                "backend": "apple_metal",
                "task_kind": "train_token_superposition_metal_cli",
                "worker_kind": "rust_metal_token_superposition_readout",
                "claimed_work_units": claimed_work_units
            }),
        );
        object.insert("artifact_write".to_string(), artifact_write);
        object.insert("raw_gate_promotion_decision".to_string(), raw_promotion);
        object.insert(
            "promotion_decision".to_string(),
            serde_json::json!({
                "promote_to_training_lane": false,
                "status": "not_promoted_keep_as_native_contract_evidence",
                "reason": "Mac Rust/Metal token-superposition report contract exists, but full native parity, scheduler routing, and public-transfer gates remain locked.",
                "artifact": ""
            }),
        );
        object.insert(
            "model_promotion_allowed".to_string(),
            serde_json::json!(false),
        );
        object.insert(
            "train_token_superposition_parity_claim_allowed".to_string(),
            serde_json::json!(false),
        );
        object.insert(
            "full_cli_parity_claim_allowed".to_string(),
            serde_json::json!(false),
        );
        object.insert("teacher_used".to_string(), serde_json::json!(false));
        object.insert("public_training_rows".to_string(), serde_json::json!(0));
        object.insert("external_inference_calls".to_string(), serde_json::json!(0));
        object.insert(
            "report_contract".to_string(),
            serde_json::json!({
                "matches_train_token_superposition_cli_surface": true,
                "mirrors_command": "train-token-superposition-mlx",
                "scheduler_routing_enabled": false,
                "scheduler_routing_blocker": "Do not route production scheduler work to Metal until full parity, route policy, and rollback guardrails are proven.",
                "python_mlx_bridge_used": false,
                "native_readout_subpath": "readout_bag_sgd_samples_kernel + readout_sgd_samples_kernel + linear_readout_logits_kernel"
            }),
        );
        object.insert(
            "guardrails".to_string(),
            serde_json::json!({
                "no_public_calibration": true,
                "no_public_training_rows": true,
                "no_teacher": true,
                "no_external_inference": true,
                "no_fallback_returns": true,
                "does_not_claim_full_kernel_parity": true,
                "does_not_claim_training_lane_parity": true,
                "does_not_route_scheduler_to_metal": true,
                "promotion_locked_by_macos_contract": true
            }),
        );
    }
}

fn standalone_metal_artifact_write_from_args(args: &serde_json::Value) -> serde_json::Value {
    readout_artifact_write_from_args(
        args,
        "structured_cgs_vsa_metal_readout",
        "train_standalone_parity_claim_allowed",
        "Canonical readout weights/bias artifact written for Metal standalone path; scheduler routing and model promotion remain separately gated.",
    )
}

fn token_superposition_artifact_write_from_args(args: &serde_json::Value) -> serde_json::Value {
    readout_artifact_write_from_args(
        args,
        "metal_token_superposition_readout_private_residual_train_eval",
        "train_token_superposition_parity_claim_allowed",
        "Canonical readout weights/bias artifact written for Metal token-superposition path; scheduler routing and model promotion remain separately gated.",
    )
}

fn readout_artifact_write_from_args(
    args: &serde_json::Value,
    expected_feature_set: &str,
    parity_claim_key: &str,
    reason: &str,
) -> serde_json::Value {
    let path = match args
        .get("model_out")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(path) => path,
        None => {
            return serde_json::json!({
                "attempted": false,
                "reason": "no_artifact_path_requested"
            })
        }
    };
    let artifact_text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            return serde_json::json!({
                "attempted": true,
                "path": path,
                "kind": "canonical_readout_artifact",
                "schema": "symliquid_core::benchmarks::ReadoutArtifact",
                "production_checkpoint_compatible": false,
                "promotion_allowed": false,
                "error": format!("artifact_read_failed:{error}")
            })
        }
    };
    let artifact = match serde_json::from_str::<serde_json::Value>(&artifact_text) {
        Ok(value) => value,
        Err(error) => {
            return serde_json::json!({
                "attempted": true,
                "path": path,
                "kind": "canonical_readout_artifact",
                "schema": "symliquid_core::benchmarks::ReadoutArtifact",
                "production_checkpoint_compatible": false,
                "promotion_allowed": false,
                "error": format!("artifact_json_failed:{error}")
            })
        }
    };
    let hv_dim = artifact
        .get("hv_dim")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let output_dim = artifact
        .get("output_dim")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let weights_written = artifact
        .get("weights")
        .and_then(|value| value.as_array())
        .map(|rows| rows.len())
        .unwrap_or(0);
    let bias_written = artifact
        .get("bias")
        .and_then(|value| value.as_array())
        .map(|rows| rows.len())
        .unwrap_or(0);
    let labels_written = artifact
        .get("labels")
        .and_then(|value| value.as_array())
        .map(|rows| rows.len())
        .unwrap_or(0);
    let feature_set = artifact
        .get("feature_set")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    let shape_valid = hv_dim > 0
        && output_dim > 0
        && weights_written as u64 == hv_dim.saturating_mul(output_dim)
        && bias_written as u64 == output_dim
        && labels_written as u64 == output_dim;
    let feature_set_valid = feature_set == expected_feature_set;
    let mut payload = serde_json::json!({
        "attempted": true,
        "path": path,
        "kind": "canonical_readout_artifact",
        "schema": "symliquid_core::benchmarks::ReadoutArtifact",
        "production_checkpoint_compatible": shape_valid && feature_set_valid,
        "promotion_allowed": false,
        "weights_written": weights_written,
        "bias_written": bias_written,
        "labels_written": labels_written,
        "hv_dim": hv_dim,
        "output_dim": output_dim,
        "feature_set": feature_set,
        "reason": reason
    });
    if let Some(object) = payload.as_object_mut() {
        object.insert(parity_claim_key.to_string(), serde_json::json!(false));
    }
    payload
}

fn get_json_f64(value: &serde_json::Value, path: &[&str]) -> serde_json::Value {
    let mut current = value;
    for key in path {
        match current.get(*key) {
            Some(next) => current = next,
            None => return serde_json::Value::Null,
        }
    }
    current.clone()
}

fn get_json_u64(value: &serde_json::Value, path: &[&str]) -> Option<u64> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_u64()
}

fn get_json_f64_number(value: &serde_json::Value, path: &[&str]) -> Option<f64> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_f64()
}

fn mean_std_f64(values: &[f64]) -> (f64, f64) {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return (0.0, 0.0);
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let variance = finite
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / finite.len() as f64;
    (mean, variance.sqrt())
}

fn metal_sweep_f32_slug(value: f32) -> String {
    format!("{value:.6}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .replace('-', "m")
        .replace('.', "p")
}

fn metal_sweep_case_offset(
    seed: u64,
    state_epochs: usize,
    state_lr: f32,
    run_index: usize,
    base: usize,
) -> usize {
    let seed_part = (seed % 1_000_000) as usize;
    let lr_part = (state_lr.max(0.0) * 1_000_000.0).round() as usize;
    base.saturating_add(seed_part.saturating_mul(101))
        .saturating_add(state_epochs.saturating_mul(1_003))
        .saturating_add(lr_part)
        .saturating_add(run_index.saturating_mul(17))
}

fn compact_metal_sweep_child(child: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "ok": child.get("ok").cloned().unwrap_or(serde_json::Value::Null),
        "state": child.get("state").cloned().unwrap_or(serde_json::Value::Null),
        "policy": child.get("policy").cloned().unwrap_or(serde_json::Value::Null),
        "command": child.get("command").cloned().unwrap_or(serde_json::Value::Null),
        "backend": child.get("backend").cloned().unwrap_or(serde_json::Value::Null),
        "implementation": child.get("implementation").cloned().unwrap_or(serde_json::Value::Null),
        "parity_for": child.get("parity_for").cloned().unwrap_or(serde_json::Value::Null),
        "metrics": child.get("metrics").cloned().unwrap_or_else(|| serde_json::json!({})),
        "work_receipt": child.get("work_receipt").cloned().unwrap_or_else(|| serde_json::json!({})),
        "artifact_write": child.get("artifact_write").cloned().unwrap_or_else(|| serde_json::json!({})),
        "sweep_child": child.get("sweep_child").cloned().unwrap_or_else(|| serde_json::json!({})),
        "guardrails": child.get("guardrails").cloned().unwrap_or_else(|| serde_json::json!({})),
        "model_promotion_allowed": child.get("model_promotion_allowed").cloned().unwrap_or(serde_json::Value::Null),
        "train_rollout_parity_claim_allowed": child.get("train_rollout_parity_claim_allowed").cloned().unwrap_or(serde_json::Value::Null),
        "full_cli_parity_claim_allowed": child.get("full_cli_parity_claim_allowed").cloned().unwrap_or(serde_json::Value::Null),
        "external_inference_calls": child.get("external_inference_calls").cloned().unwrap_or(serde_json::Value::Null),
        "teacher_used": child.get("teacher_used").cloned().unwrap_or(serde_json::Value::Null),
        "public_training_rows": child.get("public_training_rows").cloned().unwrap_or(serde_json::Value::Null)
    })
}

fn chrono_like_utc_now() -> String {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!(
        "unix:{}.{:09}Z",
        duration.as_secs(),
        duration.subsec_nanos()
    )
}

fn mlx_bridge_python() -> String {
    if let Ok(value) = std::env::var("THESEUS_MLX_PYTHON") {
        if !value.trim().is_empty() {
            return value;
        }
    }
    let source_venv = Path::new(".venv-puffer").join("bin").join("python");
    if source_venv.exists() {
        return source_venv.to_string_lossy().to_string();
    }
    let installed = std::env::var("HOME")
        .ok()
        .map(|home| {
            Path::new(&home)
                .join("Library")
                .join("Application Support")
                .join("Project Theseus Hive")
                .join("app")
                .join("current")
                .join(".venv-puffer")
                .join("bin")
                .join("python")
        })
        .filter(|path| path.exists());
    if let Some(path) = installed {
        return path.to_string_lossy().to_string();
    }
    "python3".to_string()
}

fn run_mlx_bridge(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let python = mlx_bridge_python();
    let status = std::process::Command::new(&python).args(&args).status()?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "MLX bridge command failed with status {status}: {} {}",
            python,
            args.join(" ")
        )
        .into())
    }
}

fn run_with_large_stack<F>(name: &'static str, f: F) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnOnce() -> Result<(), Box<dyn std::error::Error>> + Send + 'static,
{
    let handle = std::thread::Builder::new()
        .name(format!("theseus-{name}"))
        .stack_size(large_stack_size_bytes())
        .spawn(move || f().map_err(|err| err.to_string()))?;
    match handle.join() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(message)) => Err(message.into()),
        Err(_) => Err(format!("{name} panicked while running on the large-stack worker").into()),
    }
}

fn main() {
    let handle = std::thread::Builder::new()
        .name("theseus-cli-main".to_string())
        .stack_size(large_stack_size_bytes())
        .spawn(|| cli_main().map_err(|err| err.to_string()));
    match handle {
        Ok(join) => match join.join() {
            Ok(Ok(())) => {}
            Ok(Err(message)) => {
                eprintln!("{message}");
                std::process::exit(1);
            }
            Err(_) => {
                eprintln!("symliquid-cli panicked while running on the large-stack CLI thread");
                std::process::exit(1);
            }
        },
        Err(err) => {
            eprintln!("failed to start large-stack CLI thread: {err}");
            std::process::exit(1);
        }
    }
}

fn cli_main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::RoleFiller {
            steps,
            seed,
            variant,
        } => {
            let report = run_role(RoleFillerConfig {
                steps,
                seed,
                ablations: AblationConfig::named(&variant),
                variant,
                ..RoleFillerConfig::default()
            })?;
            println!("{}", format_report(&report));
        }
        Command::DelayedRecall {
            steps,
            seed,
            variant,
        } => {
            let report = run_delayed(DelayedRecallConfig {
                steps,
                seed,
                ablations: AblationConfig::named(&variant),
                variant,
                ..DelayedRecallConfig::default()
            })?;
            println!("{}", format_report(&report));
        }
        Command::ActiveClassification {
            episodes,
            seed,
            variant,
        } => {
            let report = run_active(ActiveClassificationConfig {
                episodes,
                seed,
                ablations: AblationConfig::named(&variant),
                variant,
                ..ActiveClassificationConfig::default()
            })?;
            println!("{}", format_report(&report));
        }
        Command::Gridworld {
            episodes,
            seed,
            variant,
        } => {
            let report = run_gridworld(ActiveGridworldConfig {
                episodes,
                seed,
                ablations: AblationConfig::named(&variant),
                variant,
                ..ActiveGridworldConfig::default()
            })?;
            println!("{}", format_report(&report));
        }
        Command::Ablations { task, steps, seed } => {
            let reports = run_ablation_suite(&task, steps, seed)?;
            println!("Task: {task}");
            println!("Seed: {seed}");
            println!("Device: cpu\n");
            println!("{}", format_table(&reports));
        }
        Command::BenchmarkSnapshot {
            seed,
            cases_per_task,
            out,
        } => {
            let suite = generate_cgs_hard_suite(seed, cases_per_task);
            write_suite_json(&out, &suite)?;
            println!(
                "Wrote benchmark suite '{}' with {} cases to {}",
                suite.name,
                suite.cases.len(),
                out
            );
        }
        Command::BenchmarkSymliquid {
            suite,
            out,
            model_id,
            hybrid,
        } => {
            let suite = read_suite_json(&suite)?;
            let report = run_symliquid_suite(&suite, &model_id, hybrid);
            write_report(&out, &report)?;
            println!("{}", format_summary(&report.summary));
            println!("Wrote report to {out}");
        }
        Command::BenchmarkTemplate {
            suite,
            out,
            model_id,
            hybrid,
        } => {
            let suite = read_suite_json(&suite)?;
            write_request_template(&out, &suite, &model_id, hybrid)?;
            println!(
                "Wrote {} local request templates for model '{}' to {}",
                suite.cases.len(),
                model_id,
                out
            );
        }
        Command::BenchmarkScore {
            suite,
            responses,
            out,
            model_id,
            mode,
        } => {
            let suite = read_suite_json(&suite)?;
            let mode = parse_run_mode(&mode)?;
            let report = score_responses(&suite, responses, &model_id, mode)?;
            write_report(&out, &report)?;
            println!("{}", format_summary(&report.summary));
            println!("Wrote report to {out}");
        }
        Command::BenchmarkCompare {
            baseline,
            candidate,
            out,
        } => {
            let baseline_report = read_report_or_train_eval(&baseline)?;
            let candidate_report = read_report_or_train_eval(&candidate)?;
            let comparison =
                compare_reports(&baseline, &baseline_report, &candidate, &candidate_report);
            write_comparison(&out, &comparison)?;
            println!("{}", format_comparison(&comparison));
            println!("Wrote comparison to {out}");
        }
        Command::BenchmarkBreakdown {
            suite,
            report,
            out,
            group_by,
        } => {
            let suite = read_suite_json(&suite)?;
            let report = read_report_or_train_eval(&report)?;
            write_breakdown_csv(&out, &suite, &report, &group_by)?;
            println!("Wrote benchmark breakdown grouped by '{group_by}' to {out}");
        }
        Command::TrainStandalone {
            train_seed,
            eval_seed,
            cases_per_task,
            epochs,
            batch_size,
            hv_dim,
            lr,
            symbolic_fallback,
            model_out,
            out,
        } => {
            let report = train_standalone_symliquid(StandaloneTrainConfig {
                train_seed,
                eval_seed,
                cases_per_task,
                epochs,
                batch_size,
                hv_dim,
                lr,
                symbolic_fallback,
                artifact_path: model_out,
            })?;
            write_standalone_train_report(&out, &report)?;
            println!(
                "{}",
                symliquid_core::benchmarks::format_standalone_train_report(&report)
            );
            println!("Wrote standalone training report to {out}");
        }
        Command::TrainStandaloneCuda {
            train_seed,
            eval_seed,
            cases_per_task,
            epochs,
            samples_per_launch,
            hv_dim,
            lr,
            symbolic_fallback,
            model_out,
            out,
        } => {
            #[cfg(feature = "cuda")]
            {
                let report = symliquid_cuda::readout_cuda::train_standalone_symliquid_cuda(
                    StandaloneTrainConfig {
                        train_seed,
                        eval_seed,
                        cases_per_task,
                        epochs,
                        batch_size: 1,
                        hv_dim,
                        lr,
                        symbolic_fallback,
                        artifact_path: model_out,
                    },
                    samples_per_launch,
                )?;
                write_standalone_train_report(&out, &report)?;
                println!(
                    "{}",
                    symliquid_core::benchmarks::format_standalone_train_report(&report)
                );
                println!(
                    "Wrote CUDA standalone training report to {out} (samples_per_launch={samples_per_launch})"
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (
                    train_seed,
                    eval_seed,
                    cases_per_task,
                    epochs,
                    samples_per_launch,
                    hv_dim,
                    lr,
                    symbolic_fallback,
                    model_out,
                    out,
                );
                return Err(
                    "train-standalone-cuda requires building symliquid-cli with --features cuda"
                        .into(),
                );
            }
        }
        Command::TrainStandaloneMlx {
            train_seed,
            eval_seed,
            cases_per_task,
            epochs,
            batch_size,
            samples_per_launch,
            hv_dim,
            lr,
            symbolic_fallback,
            model_out,
            train_input,
            eval_input,
            profile,
            out,
        } => {
            let mut args = mlx_bridge_args("train-standalone-mlx");
            push_cli_arg(&mut args, "--train-seed", train_seed);
            push_cli_arg(&mut args, "--eval-seed", eval_seed);
            push_cli_arg(&mut args, "--cases-per-task", cases_per_task);
            push_cli_arg(&mut args, "--epochs", epochs);
            push_cli_arg(&mut args, "--batch-size", batch_size);
            push_cli_arg(&mut args, "--samples-per-launch", samples_per_launch);
            push_cli_arg(&mut args, "--hv-dim", hv_dim);
            push_cli_arg(&mut args, "--lr", lr);
            if symbolic_fallback {
                args.push("--symbolic-fallback".to_string());
            }
            if let Some(model_out) = model_out {
                push_cli_arg(&mut args, "--model-out", model_out);
            }
            push_cli_arg(&mut args, "--train-input", train_input);
            push_cli_arg(&mut args, "--eval-input", eval_input);
            push_cli_arg(&mut args, "--profile", profile);
            push_cli_arg(&mut args, "--out", out);
            run_mlx_bridge(args)?;
        }
        Command::TrainStandaloneMetal {
            train_seed,
            eval_seed,
            cases_per_task,
            epochs,
            samples_per_launch,
            hv_dim,
            lr,
            symbolic_fallback,
            model_out,
            out,
        } => {
            let report = symliquid_metal::train_standalone_symliquid_metal(
                StandaloneTrainConfig {
                    train_seed,
                    eval_seed,
                    cases_per_task,
                    epochs,
                    batch_size: 1,
                    hv_dim,
                    lr,
                    symbolic_fallback,
                    artifact_path: model_out.clone(),
                },
                samples_per_launch,
            )?;
            let args = serde_json::json!({
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "cases_per_task": cases_per_task,
                "epochs": epochs,
                "samples_per_launch": samples_per_launch,
                "hv_dim": hv_dim,
                "lr": lr,
                "symbolic_fallback": symbolic_fallback,
                "model_out": model_out
            });
            let mut value = serde_json::to_value(&report)?;
            annotate_train_standalone_metal_report(&mut value, args);
            write_json_value(&out, &value)?;
            println!(
                "{}",
                symliquid_core::benchmarks::format_standalone_train_report(&report)
            );
            println!(
                "Wrote Metal standalone training report to {out} (samples_per_launch={samples_per_launch})"
            );
        }
        Command::TrainRolloutCuda {
            train_seed,
            eval_seed,
            cases_per_task,
            epochs,
            state_epochs,
            state_lr,
            samples_per_launch,
            probe_cases_per_task,
            rollout_batch,
            obs_dim,
            hidden_dim,
            reservoir_dim,
            hv_dim,
            seq_len,
            lr,
            model_out,
            out,
        } => {
            #[cfg(feature = "cuda")]
            {
                let report = symliquid_cuda::rollout_cuda::train_standalone_rollout_cuda(
                    StandaloneTrainConfig {
                        train_seed,
                        eval_seed,
                        cases_per_task,
                        epochs,
                        batch_size: 1,
                        hv_dim,
                        lr,
                        symbolic_fallback: false,
                        artifact_path: model_out,
                    },
                    symliquid_cuda::rollout_cuda::RolloutFeatureConfig {
                        obs_dim,
                        hidden_dim,
                        reservoir_dim,
                        hv_dim,
                        seq_len,
                        rollout_batch,
                        dt: 0.1,
                        alpha: 0.25,
                        memory_decay: 0.98,
                    },
                    samples_per_launch,
                    state_epochs,
                    state_lr,
                    probe_cases_per_task,
                )?;
                write_standalone_train_report(&out, &report)?;
                println!(
                    "{}",
                    symliquid_core::benchmarks::format_standalone_train_report(&report)
                );
                println!(
                    "Wrote CUDA rollout training report to {out} (rollout_batch={rollout_batch}, samples_per_launch={samples_per_launch})"
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (
                    train_seed,
                    eval_seed,
                    cases_per_task,
                    epochs,
                    state_epochs,
                    state_lr,
                    samples_per_launch,
                    probe_cases_per_task,
                    rollout_batch,
                    obs_dim,
                    hidden_dim,
                    reservoir_dim,
                    hv_dim,
                    seq_len,
                    lr,
                    model_out,
                    out,
                );
                return Err(
                    "train-rollout-cuda requires building symliquid-cli with --features cuda"
                        .into(),
                );
            }
        }
        Command::TrainRolloutMlx {
            train_seed,
            eval_seed,
            cases_per_task,
            epochs,
            state_epochs,
            state_lr,
            samples_per_launch,
            probe_cases_per_task,
            rollout_batch,
            obs_dim,
            hidden_dim,
            reservoir_dim,
            hv_dim,
            seq_len,
            lr,
            model_out,
            profile,
            out,
        } => {
            let mut args = mlx_bridge_args("train-rollout-mlx");
            push_cli_arg(&mut args, "--train-seed", train_seed);
            push_cli_arg(&mut args, "--eval-seed", eval_seed);
            push_cli_arg(&mut args, "--cases-per-task", cases_per_task);
            push_cli_arg(&mut args, "--epochs", epochs);
            push_cli_arg(&mut args, "--state-epochs", state_epochs);
            push_cli_arg(&mut args, "--state-lr", state_lr);
            push_cli_arg(&mut args, "--samples-per-launch", samples_per_launch);
            push_cli_arg(&mut args, "--probe-cases-per-task", probe_cases_per_task);
            push_cli_arg(&mut args, "--rollout-batch", rollout_batch);
            push_cli_arg(&mut args, "--obs-dim", obs_dim);
            push_cli_arg(&mut args, "--hidden-dim", hidden_dim);
            push_cli_arg(&mut args, "--reservoir-dim", reservoir_dim);
            push_cli_arg(&mut args, "--hv-dim", hv_dim);
            push_cli_arg(&mut args, "--seq-len", seq_len);
            push_cli_arg(&mut args, "--lr", lr);
            if let Some(model_out) = model_out {
                push_cli_arg(&mut args, "--model-out", model_out);
            }
            push_cli_arg(&mut args, "--profile", profile);
            push_cli_arg(&mut args, "--out", out);
            run_mlx_bridge(args)?;
        }
        Command::TrainRolloutMetal {
            train_seed,
            eval_seed,
            cases_per_task,
            epochs,
            state_epochs,
            state_lr,
            samples_per_launch,
            probe_cases_per_task,
            rollout_batch,
            obs_dim,
            hidden_dim,
            reservoir_dim,
            hv_dim,
            seq_len,
            lr,
            output_dim,
            tolerance,
            model_out,
            profile,
            out,
        } => {
            let model_out_for_args = model_out.clone();
            let profile_for_args = profile.clone();
            let out_for_args = out.clone();
            let mut report = symliquid_metal::rollout_metal_train_path_proof_report(
                &symliquid_metal::RolloutMetalProofConfig {
                    cases: cases_per_task.max(1),
                    batch: rollout_batch.max(1),
                    steps: seq_len.max(1),
                    obs_dim: obs_dim.max(1),
                    hidden_dim: hidden_dim.max(1),
                    reservoir_dim: reservoir_dim.max(1),
                    hv_dim: hv_dim.max(8),
                    output_dim: output_dim.max(1),
                    readout_epochs: epochs.max(1),
                    readout_lr: lr,
                    samples_per_launch: samples_per_launch.max(1),
                    tolerance,
                    artifact_path: model_out.clone(),
                    ..symliquid_metal::RolloutMetalProofConfig::default()
                },
            );
            annotate_train_rollout_metal_report(
                &mut report,
                serde_json::json!({
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "cases_per_task": cases_per_task,
                    "epochs": epochs,
                    "state_epochs": state_epochs,
                    "state_lr": state_lr,
                    "samples_per_launch": samples_per_launch,
                    "probe_cases_per_task": probe_cases_per_task,
                    "rollout_batch": rollout_batch,
                    "obs_dim": obs_dim,
                    "hidden_dim": hidden_dim,
                    "reservoir_dim": reservoir_dim,
                    "hv_dim": hv_dim,
                    "seq_len": seq_len,
                    "lr": lr,
                    "output_dim": output_dim,
                    "tolerance": tolerance,
                    "model_out": model_out_for_args,
                    "profile": profile_for_args,
                    "out": out_for_args
                }),
            );
            write_json_value(&out, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            let ok = report
                .get("ok")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            if !ok {
                return Err(format!("train-rollout-metal failed; wrote report to {out}").into());
            }
            println!("Wrote Metal rollout training report to {out}");
        }
        Command::RolloutMetalProof {
            batch,
            steps,
            obs_dim,
            hidden_dim,
            reservoir_dim,
            hv_dim,
            tolerance,
            out,
        } => {
            let report = symliquid_metal::rollout_metal_proof_report(
                &symliquid_metal::RolloutMetalProofConfig {
                    cases: symliquid_metal::RolloutMetalProofConfig::default().cases,
                    batch,
                    steps,
                    obs_dim,
                    hidden_dim,
                    reservoir_dim,
                    hv_dim,
                    tolerance,
                    ..symliquid_metal::RolloutMetalProofConfig::default()
                },
            );
            write_json_value(&out, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            let ok = report
                .get("ok")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            if !ok {
                return Err(format!("rollout-metal-proof failed; wrote report to {out}").into());
            }
            println!("Wrote Metal rollout proof report to {out}");
        }
        Command::RolloutMetalFeatureProof {
            cases,
            batch,
            steps,
            obs_dim,
            hidden_dim,
            reservoir_dim,
            hv_dim,
            tolerance,
            out,
        } => {
            let report = symliquid_metal::rollout_metal_feature_proof_report(
                &symliquid_metal::RolloutMetalProofConfig {
                    cases,
                    batch,
                    steps,
                    obs_dim,
                    hidden_dim,
                    reservoir_dim,
                    hv_dim,
                    tolerance,
                    ..symliquid_metal::RolloutMetalProofConfig::default()
                },
            );
            write_json_value(&out, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            let ok = report
                .get("ok")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            if !ok {
                return Err(
                    format!("rollout-metal-feature-proof failed; wrote report to {out}").into(),
                );
            }
            println!("Wrote Metal rollout feature proof report to {out}");
        }
        Command::RolloutMetalReadoutProof {
            cases,
            batch,
            steps,
            obs_dim,
            hidden_dim,
            reservoir_dim,
            hv_dim,
            output_dim,
            tolerance,
            out,
        } => {
            let report = symliquid_metal::rollout_metal_readout_proof_report(
                &symliquid_metal::RolloutMetalProofConfig {
                    cases,
                    batch,
                    steps,
                    obs_dim,
                    hidden_dim,
                    reservoir_dim,
                    hv_dim,
                    output_dim,
                    tolerance,
                    ..symliquid_metal::RolloutMetalProofConfig::default()
                },
            );
            write_json_value(&out, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            let ok = report
                .get("ok")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            if !ok {
                return Err(
                    format!("rollout-metal-readout-proof failed; wrote report to {out}").into(),
                );
            }
            println!("Wrote Metal rollout readout proof report to {out}");
        }
        Command::RolloutMetalReadoutTrainingProof {
            cases,
            batch,
            steps,
            obs_dim,
            hidden_dim,
            reservoir_dim,
            hv_dim,
            output_dim,
            readout_epochs,
            readout_lr,
            samples_per_launch,
            tolerance,
            out,
        } => {
            let report = symliquid_metal::rollout_metal_readout_training_proof_report(
                &symliquid_metal::RolloutMetalProofConfig {
                    cases,
                    batch,
                    steps,
                    obs_dim,
                    hidden_dim,
                    reservoir_dim,
                    hv_dim,
                    output_dim,
                    readout_epochs,
                    readout_lr,
                    samples_per_launch,
                    tolerance,
                    ..symliquid_metal::RolloutMetalProofConfig::default()
                },
            );
            write_json_value(&out, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            let ok = report
                .get("ok")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            if !ok {
                return Err(format!(
                    "rollout-metal-readout-training-proof failed; wrote report to {out}"
                )
                .into());
            }
            println!("Wrote Metal rollout readout training proof report to {out}");
        }
        Command::RolloutMetalTrainPathProof {
            cases,
            batch,
            steps,
            obs_dim,
            hidden_dim,
            reservoir_dim,
            hv_dim,
            output_dim,
            readout_epochs,
            readout_lr,
            samples_per_launch,
            tolerance,
            out,
        } => {
            let report = symliquid_metal::rollout_metal_train_path_proof_report(
                &symliquid_metal::RolloutMetalProofConfig {
                    cases,
                    batch,
                    steps,
                    obs_dim,
                    hidden_dim,
                    reservoir_dim,
                    hv_dim,
                    output_dim,
                    readout_epochs,
                    readout_lr,
                    samples_per_launch,
                    tolerance,
                    ..symliquid_metal::RolloutMetalProofConfig::default()
                },
            );
            write_json_value(&out, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            let ok = report
                .get("ok")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            if !ok {
                return Err(format!(
                    "rollout-metal-train-path-proof failed; wrote report to {out}"
                )
                .into());
            }
            println!("Wrote Metal train path proof report to {out}");
        }
        Command::RolloutMetalStateTrainingProof {
            cases,
            batch,
            steps,
            obs_dim,
            hidden_dim,
            reservoir_dim,
            hv_dim,
            output_dim,
            state_epochs,
            state_lr,
            tolerance,
            out,
        } => {
            let report = symliquid_metal::rollout_metal_state_training_proof_report(
                &symliquid_metal::RolloutMetalProofConfig {
                    cases,
                    batch,
                    steps,
                    obs_dim,
                    hidden_dim,
                    reservoir_dim,
                    hv_dim,
                    output_dim,
                    state_epochs,
                    state_lr,
                    tolerance,
                    ..symliquid_metal::RolloutMetalProofConfig::default()
                },
            );
            write_json_value(&out, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            let ok = report
                .get("ok")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            if !ok {
                return Err(format!(
                    "rollout-metal-state-training-proof failed; wrote report to {out}"
                )
                .into());
            }
            println!("Wrote Metal rollout state-training proof report to {out}");
        }
        Command::TokenSuperpositionMetalReadoutProof {
            vocab_size,
            hv_dim,
            train_tokens,
            train_samples,
            eval_samples,
            baseline_epochs,
            bag_size,
            recovery_ratio,
            lr,
            samples_per_launch,
            tolerance,
            out,
        } => {
            let report = symliquid_metal::token_superposition_metal_readout_proof_report(
                &symliquid_metal::TokenSuperpositionMetalProofConfig {
                    vocab_size,
                    hv_dim,
                    train_tokens,
                    train_samples,
                    eval_samples,
                    baseline_epochs,
                    bag_size,
                    recovery_ratio,
                    lr,
                    samples_per_launch,
                    tolerance,
                },
            );
            write_json_value(&out, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            let ok = report
                .get("ok")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            if !ok {
                return Err(format!(
                    "token-superposition-metal-readout-proof failed; wrote report to {out}"
                )
                .into());
            }
            println!("Wrote Metal token-superposition readout proof report to {out}");
        }
        Command::TrainRolloutCudaSweep {
            train_seeds,
            eval_seed_base,
            cases_per_task,
            epochs,
            state_epochs,
            state_lrs,
            samples_per_launch,
            probe_cases_per_task,
            rollout_batch,
            obs_dim,
            hidden_dim,
            reservoir_dim,
            hv_dim,
            seq_len,
            lr,
            out,
        } => {
            #[cfg(feature = "cuda")]
            {
                let seeds = parse_seed_list(&train_seeds)?;
                let state_epoch_grid = parse_usize_list(&state_epochs)?;
                let state_lr_grid = parse_f32_list(&state_lrs)?;
                let mut runs = Vec::new();
                for (seed_idx, train_seed) in seeds.iter().copied().enumerate() {
                    let eval_seed = eval_seed_base + seed_idx as u64;
                    for state_epochs in &state_epoch_grid {
                        let state_lrs_for_epoch = if *state_epochs == 0 {
                            vec![0.0]
                        } else {
                            state_lr_grid.clone()
                        };
                        for state_lr in state_lrs_for_epoch {
                            let report =
                                symliquid_cuda::rollout_cuda::train_standalone_rollout_cuda(
                                    StandaloneTrainConfig {
                                        train_seed,
                                        eval_seed,
                                        cases_per_task,
                                        epochs,
                                        batch_size: 1,
                                        hv_dim,
                                        lr,
                                        symbolic_fallback: false,
                                        artifact_path: None,
                                    },
                                    symliquid_cuda::rollout_cuda::RolloutFeatureConfig {
                                        obs_dim,
                                        hidden_dim,
                                        reservoir_dim,
                                        hv_dim,
                                        seq_len,
                                        rollout_batch,
                                        dt: 0.1,
                                        alpha: 0.25,
                                        memory_decay: 0.98,
                                    },
                                    samples_per_launch,
                                    *state_epochs,
                                    state_lr,
                                    probe_cases_per_task,
                                )?;
                            println!(
                                "sweep run seed={} eval_seed={} state_epochs={} state_lr={:.5} accuracy={:.3} residual={:.3} accepted_state={}",
                                train_seed,
                                eval_seed,
                                state_epochs,
                                state_lr,
                                report.eval.summary.accuracy,
                                report.eval.summary.residual,
                                report
                                    .state_training
                                    .as_ref()
                                    .map(|state| state.accepted)
                                    .unwrap_or(false)
                            );
                            runs.push(report);
                        }
                    }
                }
                let report = build_rollout_cuda_sweep_report(RolloutCudaSweepBuild {
                    train_seeds: seeds,
                    eval_seed_base,
                    cases_per_task,
                    epochs,
                    state_epoch_grid,
                    state_lr_grid,
                    samples_per_launch,
                    probe_cases_per_task,
                    rollout_batch,
                    obs_dim,
                    hidden_dim,
                    reservoir_dim,
                    hv_dim,
                    seq_len,
                    lr,
                    runs,
                })?;
                write_rollout_cuda_sweep_report(&out, &report)?;
                println!("{}", format_rollout_cuda_sweep_report(&report));
                println!("Wrote CUDA rollout sweep report to {out}");
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (
                    train_seeds,
                    eval_seed_base,
                    cases_per_task,
                    epochs,
                    state_epochs,
                    state_lrs,
                    samples_per_launch,
                    probe_cases_per_task,
                    rollout_batch,
                    obs_dim,
                    hidden_dim,
                    reservoir_dim,
                    hv_dim,
                    seq_len,
                    lr,
                    out,
                );
                return Err(
                    "train-rollout-cuda-sweep requires building symliquid-cli with --features cuda"
                        .into(),
                );
            }
        }
        Command::TrainRolloutMlxSweep {
            train_seeds,
            eval_seed_base,
            cases_per_task,
            epochs,
            state_epochs,
            state_lrs,
            samples_per_launch,
            probe_cases_per_task,
            rollout_batch,
            obs_dim,
            hidden_dim,
            reservoir_dim,
            hv_dim,
            seq_len,
            lr,
            profile,
            out,
        } => {
            let mut args = mlx_bridge_args("train-rollout-mlx-sweep");
            push_cli_arg(&mut args, "--train-seeds", train_seeds);
            push_cli_arg(&mut args, "--eval-seed-base", eval_seed_base);
            push_cli_arg(&mut args, "--cases-per-task", cases_per_task);
            push_cli_arg(&mut args, "--epochs", epochs);
            push_cli_arg(&mut args, "--state-epochs", state_epochs);
            push_cli_arg(&mut args, "--state-lrs", state_lrs);
            push_cli_arg(&mut args, "--samples-per-launch", samples_per_launch);
            push_cli_arg(&mut args, "--probe-cases-per-task", probe_cases_per_task);
            push_cli_arg(&mut args, "--rollout-batch", rollout_batch);
            push_cli_arg(&mut args, "--obs-dim", obs_dim);
            push_cli_arg(&mut args, "--hidden-dim", hidden_dim);
            push_cli_arg(&mut args, "--reservoir-dim", reservoir_dim);
            push_cli_arg(&mut args, "--hv-dim", hv_dim);
            push_cli_arg(&mut args, "--seq-len", seq_len);
            push_cli_arg(&mut args, "--lr", lr);
            push_cli_arg(&mut args, "--profile", profile);
            push_cli_arg(&mut args, "--out", out);
            run_mlx_bridge(args)?;
        }
        Command::TrainRolloutMetalSweep {
            train_seeds,
            eval_seed_base,
            cases_per_task,
            epochs,
            state_epochs,
            state_lrs,
            samples_per_launch,
            probe_cases_per_task,
            rollout_batch,
            obs_dim,
            hidden_dim,
            reservoir_dim,
            hv_dim,
            seq_len,
            lr,
            output_dim,
            tolerance,
            profile,
            artifact_dir,
            out,
        } => {
            let started = std::time::Instant::now();
            let seeds = parse_seed_list(&train_seeds)?;
            let state_epoch_grid = parse_usize_list(&state_epochs)?;
            let state_lr_grid = parse_f32_list(&state_lrs)?;
            fs::create_dir_all(&artifact_dir)?;
            let mut runs = Vec::new();
            let mut eval_accuracies = Vec::new();
            let mut eval_losses = Vec::new();
            let mut total_kernel_launches = 0u64;
            let mut artifact_count = 0usize;
            let mut child_ok_count = 0usize;
            let mut best_index = 0usize;
            for (seed_idx, train_seed) in seeds.iter().copied().enumerate() {
                let eval_seed = eval_seed_base + seed_idx as u64;
                for state_epochs_value in &state_epoch_grid {
                    let state_lrs_for_epoch = if *state_epochs_value == 0 {
                        vec![0.0]
                    } else {
                        state_lr_grid.clone()
                    };
                    for state_lr_value in state_lrs_for_epoch {
                        let run_index = runs.len();
                        let lr_slug = metal_sweep_f32_slug(state_lr_value);
                        let child_prefix = format!(
                            "seed{train_seed}_eval{eval_seed}_state{state_epochs_value}_lr{lr_slug}"
                        );
                        let artifact_path = format!("{artifact_dir}/readout_{child_prefix}.json");
                        let child_report_path =
                            format!("{artifact_dir}/report_{child_prefix}.json");
                        let train_case_offset = metal_sweep_case_offset(
                            train_seed,
                            *state_epochs_value,
                            state_lr_value,
                            run_index,
                            0,
                        );
                        let eval_case_offset = metal_sweep_case_offset(
                            eval_seed,
                            *state_epochs_value,
                            state_lr_value,
                            run_index,
                            10_000,
                        );
                        let mut child = symliquid_metal::rollout_metal_train_path_proof_report(
                            &symliquid_metal::RolloutMetalProofConfig {
                                cases: cases_per_task.max(1),
                                batch: rollout_batch.max(1),
                                steps: seq_len.max(1),
                                obs_dim: obs_dim.max(1),
                                hidden_dim: hidden_dim.max(1),
                                reservoir_dim: reservoir_dim.max(1),
                                hv_dim: hv_dim.max(8),
                                output_dim: output_dim.max(1),
                                readout_epochs: epochs.max(1),
                                readout_lr: lr,
                                samples_per_launch: samples_per_launch.max(1),
                                tolerance,
                                artifact_path: Some(artifact_path.clone()),
                                train_case_offset,
                                eval_case_offset,
                                ..symliquid_metal::RolloutMetalProofConfig::default()
                            },
                        );
                        annotate_train_rollout_metal_report(
                            &mut child,
                            serde_json::json!({
                                "train_seed": train_seed,
                                "eval_seed": eval_seed,
                                "cases_per_task": cases_per_task,
                                "epochs": epochs,
                                "state_epochs": state_epochs_value,
                                "state_lr": state_lr_value,
                                "samples_per_launch": samples_per_launch,
                                "probe_cases_per_task": probe_cases_per_task,
                                "rollout_batch": rollout_batch,
                                "obs_dim": obs_dim,
                                "hidden_dim": hidden_dim,
                                "reservoir_dim": reservoir_dim,
                                "hv_dim": hv_dim,
                                "seq_len": seq_len,
                                "lr": lr,
                                "output_dim": output_dim,
                                "tolerance": tolerance,
                                "model_out": artifact_path,
                                "profile": profile,
                                "out": child_report_path,
                                "train_case_offset": train_case_offset,
                                "eval_case_offset": eval_case_offset
                            }),
                        );
                        if let Some(object) = child.as_object_mut() {
                            object.insert(
                                "sweep_child".to_string(),
                                serde_json::json!({
                                    "parent_command": "train-rollout-metal-sweep",
                                    "parity_for": "train-rollout-cuda-sweep",
                                    "run_index": run_index,
                                    "train_seed": train_seed,
                                    "eval_seed": eval_seed,
                                    "state_epochs_requested": state_epochs_value,
                                    "state_lr_requested": state_lr_value,
                                    "train_case_offset": train_case_offset,
                                    "eval_case_offset": eval_case_offset,
                                    "child_report_path": child_report_path,
                                    "artifact_path": artifact_path,
                                    "state_training_native_ported": false,
                                    "state_training_semantics": "CUDA state-training grid is recorded and used to vary private synthetic offsets; this guarded Metal sweep proves rollout/readout hot-loop execution only."
                                }),
                            );
                        }
                        write_json_value(&child_report_path, &child)?;
                        let child_ok = child
                            .get("ok")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(false);
                        if child_ok {
                            child_ok_count += 1;
                        }
                        if child
                            .get("artifact_write")
                            .and_then(|value| value.get("production_checkpoint_compatible"))
                            .and_then(|value| value.as_bool())
                            .unwrap_or(false)
                        {
                            artifact_count += 1;
                        }
                        total_kernel_launches = total_kernel_launches.saturating_add(
                            child
                                .get("kernel_launches")
                                .and_then(|value| value.as_u64())
                                .unwrap_or(0),
                        );
                        let eval_accuracy =
                            get_json_f64_number(&child, &["metrics", "eval_accuracy"])
                                .unwrap_or(0.0);
                        let eval_loss = get_json_f64_number(&child, &["metrics", "eval_loss"])
                            .unwrap_or(f64::INFINITY);
                        eval_accuracies.push(eval_accuracy);
                        if eval_loss.is_finite() {
                            eval_losses.push(eval_loss);
                        }
                        let best_accuracy = runs
                            .get(best_index)
                            .and_then(|row: &serde_json::Value| {
                                get_json_f64_number(row, &["metrics", "eval_accuracy"])
                            })
                            .unwrap_or(f64::NEG_INFINITY);
                        let best_loss = runs
                            .get(best_index)
                            .and_then(|row: &serde_json::Value| {
                                get_json_f64_number(row, &["metrics", "eval_loss"])
                            })
                            .unwrap_or(f64::INFINITY);
                        if runs.is_empty()
                            || eval_accuracy > best_accuracy
                            || ((eval_accuracy - best_accuracy).abs() <= f64::EPSILON
                                && eval_loss < best_loss)
                        {
                            best_index = run_index;
                        }
                        runs.push(child);
                    }
                }
            }
            if runs.is_empty() {
                return Err("train-rollout-metal-sweep produced no child runs".into());
            }
            let (mean_eval_accuracy, std_eval_accuracy) = mean_std_f64(&eval_accuracies);
            let (mean_eval_loss, std_eval_loss) = mean_std_f64(&eval_losses);
            let best_run = runs
                .get(best_index)
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            let ok = child_ok_count == runs.len() && artifact_count == runs.len();
            let report = serde_json::json!({
                "ok": ok,
                "policy": "project_theseus_macos_metal_rollout_sweep_v0",
                "created_utc": chrono_like_utc_now(),
                "trigger_state": if ok { "GREEN" } else { "RED" },
                "state": if ok { "GREEN" } else { "RED" },
                "command": "train-rollout-metal-sweep",
                "backend": "apple_metal",
                "implementation": "rust_metal_rollout_sweep_guarded_proof",
                "parity_for": "train-rollout-cuda-sweep",
                "score_semantics": "Guarded Mac-native sweep proof only. It runs bounded private synthetic rollout/readout Metal children over the sweep grid and does not claim CUDA state-training parity, production scheduler routing, public transfer, or model promotion.",
                "args": {
                    "train_seeds": seeds,
                    "eval_seed_base": eval_seed_base,
                    "cases_per_task": cases_per_task,
                    "epochs": epochs,
                    "state_epoch_grid": state_epoch_grid,
                    "state_lr_grid": state_lr_grid,
                    "samples_per_launch": samples_per_launch,
                    "probe_cases_per_task": probe_cases_per_task,
                    "rollout_batch": rollout_batch,
                    "obs_dim": obs_dim,
                    "hidden_dim": hidden_dim,
                    "reservoir_dim": reservoir_dim,
                    "hv_dim": hv_dim,
                    "seq_len": seq_len,
                    "lr": lr,
                    "output_dim": output_dim,
                    "tolerance": tolerance,
                    "profile": profile,
                    "artifact_dir": artifact_dir,
                    "out": out
                },
                "summary": {
                    "run_count": runs.len(),
                    "child_ok_count": child_ok_count,
                    "artifact_count": artifact_count,
                    "best_index": best_index,
                    "mean_eval_accuracy": mean_eval_accuracy,
                    "std_eval_accuracy": std_eval_accuracy,
                    "mean_eval_loss": mean_eval_loss,
                    "std_eval_loss": std_eval_loss,
                    "total_kernel_launches": total_kernel_launches,
                    "report_wall_ms": started.elapsed().as_secs_f64() * 1000.0,
                    "state_training_native_ported": false,
                    "cuda_state_training_parity_claim_allowed": false,
                    "production_scheduler_routing_enabled": false,
                    "model_promotion_allowed": false,
                    "train_rollout_sweep_parity_claim_allowed": false,
                    "native_hot_loop_parity_claim_allowed": false,
                    "external_inference_calls": 0,
                    "teacher_used": false,
                    "public_training_rows": 0,
                    "fallback_returns": 0
                },
                "metrics": {
                    "run_count": runs.len(),
                    "best_eval_accuracy": get_json_f64_number(&best_run, &["metrics", "eval_accuracy"]),
                    "best_eval_loss": get_json_f64_number(&best_run, &["metrics", "eval_loss"]),
                    "mean_eval_accuracy": mean_eval_accuracy,
                    "std_eval_accuracy": std_eval_accuracy,
                    "mean_eval_loss": mean_eval_loss,
                    "std_eval_loss": std_eval_loss,
                    "kernel_launches": total_kernel_launches
                },
                "best_run": {
                    "index": best_index,
                    "ok": best_run.get("ok").cloned().unwrap_or(serde_json::Value::Null),
                    "child_report_path": best_run.pointer("/sweep_child/child_report_path").cloned().unwrap_or(serde_json::Value::Null),
                    "artifact_path": best_run.pointer("/sweep_child/artifact_path").cloned().unwrap_or(serde_json::Value::Null),
                    "train_seed": best_run.pointer("/sweep_child/train_seed").cloned().unwrap_or(serde_json::Value::Null),
                    "eval_seed": best_run.pointer("/sweep_child/eval_seed").cloned().unwrap_or(serde_json::Value::Null),
                    "state_epochs_requested": best_run.pointer("/sweep_child/state_epochs_requested").cloned().unwrap_or(serde_json::Value::Null),
                    "state_lr_requested": best_run.pointer("/sweep_child/state_lr_requested").cloned().unwrap_or(serde_json::Value::Null),
                    "metrics": best_run.get("metrics").cloned().unwrap_or_else(|| serde_json::json!({})),
                    "artifact_write": best_run.get("artifact_write").cloned().unwrap_or_else(|| serde_json::json!({}))
                },
                "children": runs.iter().map(compact_metal_sweep_child).collect::<Vec<_>>(),
                "report_contract": {
                    "matches_train_rollout_sweep_cli_surface": true,
                    "mirrors_command": "train-rollout-mlx-sweep",
                    "child_command": "train-rollout-metal",
                    "native_readout_subpath": "rollout_state_update_kernel + readout_sgd_samples_kernel + linear_readout_logits_kernel",
                    "python_mlx_bridge_used": false,
                    "scheduler_routing_enabled": false,
                    "scheduler_routing_blocker": "Do not route production scheduler work to Metal until full parity, state-training semantics, route policy, and rollback guardrails are proven.",
                    "state_training_native_ported": false,
                    "state_training_semantics": "The CUDA sweep state-training grid is recorded and used to vary private synthetic offsets. This report does not claim CUDA state-training optimizer parity."
                },
                "work_receipt": {
                    "accepted": ok,
                    "backend": "apple_metal",
                    "task_kind": "train_rollout_metal_sweep_cli",
                    "worker_kind": "rust_metal_rollout_sweep_guarded_proof",
                    "claimed_work_units": total_kernel_launches
                },
                "promotion_decision": {
                    "promote_to_training_lane": false,
                    "status": "not_promoted_keep_as_native_sweep_evidence",
                    "reason": "Mac Rust/Metal rollout sweep proof exists, but CUDA state-training parity, production scheduler routing, public transfer, and promotion gates remain locked.",
                    "artifact": ""
                },
                "guardrails": {
                    "no_public_calibration": true,
                    "no_public_training_rows": true,
                    "no_teacher": true,
                    "no_external_inference": true,
                    "no_fallback_returns": true,
                    "does_not_claim_full_kernel_parity": true,
                    "does_not_claim_full_training_parity": true,
                    "does_not_claim_cuda_state_training_parity": true,
                    "does_not_route_scheduler_to_metal": true,
                    "scheduler_routing_enabled": false,
                    "production_scheduler_routing_enabled": false,
                    "remote_task_submitted": false,
                    "promotion_locked_by_macos_contract": true
                },
                "model_promotion_allowed": false,
                "train_rollout_sweep_parity_claim_allowed": false,
                "full_cli_parity_claim_allowed": false,
                "external_inference_calls": 0,
                "teacher_used": false,
                "public_training_rows": 0
            });
            write_json_value(&out, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !ok {
                return Err(
                    format!("train-rollout-metal-sweep failed; wrote report to {out}").into(),
                );
            }
            println!("Wrote Metal rollout sweep report to {out}");
        }
        Command::TrainTokenSuperpositionCuda {
            input,
            include_project_code,
            project_code_roots,
            train_seed,
            max_language_rows,
            max_code_files,
            max_chars_per_doc,
            max_vocab,
            hv_dim,
            train_samples,
            eval_samples,
            baseline_epochs,
            bag_sizes,
            recovery_ratios,
            lr,
            samples_per_launch,
            gate_tolerance,
            min_nominal_speedup,
            min_train_speedup,
            model_out,
            out,
        } => {
            #[cfg(feature = "cuda")]
            {
                let config = TokenSuperpositionConfig {
                    train_seed,
                    input_paths: split_string_list(&input),
                    include_project_code,
                    project_code_roots: split_string_list(&project_code_roots),
                    max_language_rows,
                    max_code_files,
                    max_chars_per_doc,
                    max_vocab,
                    hv_dim,
                    train_samples,
                    eval_samples,
                    baseline_epochs,
                    bag_sizes: parse_usize_list(&bag_sizes)?,
                    recovery_ratios: parse_f32_list(&recovery_ratios)?,
                    lr,
                    gate_tolerance,
                    min_nominal_speedup,
                    min_train_speedup,
                    artifact_path: model_out,
                };
                let report = symliquid_cuda::readout_cuda::train_token_superposition_cuda(
                    Path::new("."),
                    config,
                    samples_per_launch,
                )?;
                write_token_superposition_report(&out, &report)?;
                println!("{}", format_token_superposition_report(&report));
                println!("Wrote token superposition report to {out}");
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (
                    input,
                    include_project_code,
                    project_code_roots,
                    train_seed,
                    max_language_rows,
                    max_code_files,
                    max_chars_per_doc,
                    max_vocab,
                    hv_dim,
                    train_samples,
                    eval_samples,
                    baseline_epochs,
                    bag_sizes,
                    recovery_ratios,
                    lr,
                    samples_per_launch,
                    gate_tolerance,
                    min_nominal_speedup,
                    min_train_speedup,
                    model_out,
                    out,
                );
                return Err(
                    "train-token-superposition-cuda requires building symliquid-cli with --features cuda"
                        .into(),
                );
            }
        }
        Command::TrainTokenSuperpositionMlx {
            input,
            include_project_code,
            project_code_roots,
            train_seed,
            max_language_rows,
            max_code_files,
            max_chars_per_doc,
            max_vocab,
            hv_dim,
            train_samples,
            eval_samples,
            baseline_epochs,
            bag_sizes,
            recovery_ratios,
            lr,
            samples_per_launch,
            gate_tolerance,
            min_nominal_speedup,
            min_train_speedup,
            model_out,
            out,
        } => {
            let mut args = mlx_bridge_args("train-token-superposition-mlx");
            push_cli_arg(&mut args, "--input", input);
            if include_project_code {
                args.push("--include-project-code".to_string());
            }
            push_cli_arg(&mut args, "--project-code-roots", project_code_roots);
            push_cli_arg(&mut args, "--train-seed", train_seed);
            push_cli_arg(&mut args, "--max-language-rows", max_language_rows);
            push_cli_arg(&mut args, "--max-code-files", max_code_files);
            push_cli_arg(&mut args, "--max-chars-per-doc", max_chars_per_doc);
            push_cli_arg(&mut args, "--max-vocab", max_vocab);
            push_cli_arg(&mut args, "--hv-dim", hv_dim);
            push_cli_arg(&mut args, "--train-samples", train_samples);
            push_cli_arg(&mut args, "--eval-samples", eval_samples);
            push_cli_arg(&mut args, "--baseline-epochs", baseline_epochs);
            push_cli_arg(&mut args, "--bag-sizes", bag_sizes);
            push_cli_arg(&mut args, "--recovery-ratios", recovery_ratios);
            push_cli_arg(&mut args, "--lr", lr);
            push_cli_arg(&mut args, "--samples-per-launch", samples_per_launch);
            push_cli_arg(&mut args, "--gate-tolerance", gate_tolerance);
            push_cli_arg(&mut args, "--min-nominal-speedup", min_nominal_speedup);
            push_cli_arg(&mut args, "--min-train-speedup", min_train_speedup);
            if let Some(model_out) = model_out {
                push_cli_arg(&mut args, "--model-out", model_out);
            }
            push_cli_arg(&mut args, "--out", out);
            run_mlx_bridge(args)?;
        }
        Command::TrainTokenSuperpositionMetal {
            input,
            include_project_code,
            project_code_roots,
            train_seed,
            max_language_rows,
            max_code_files,
            max_chars_per_doc,
            max_vocab,
            hv_dim,
            train_samples,
            eval_samples,
            baseline_epochs,
            bag_sizes,
            recovery_ratios,
            lr,
            samples_per_launch,
            gate_tolerance,
            min_nominal_speedup,
            min_train_speedup,
            model_out,
            out,
        } => {
            let model_out_for_args = model_out.clone();
            let config = TokenSuperpositionConfig {
                train_seed,
                input_paths: split_string_list(&input),
                include_project_code,
                project_code_roots: split_string_list(&project_code_roots),
                max_language_rows,
                max_code_files,
                max_chars_per_doc,
                max_vocab,
                hv_dim,
                train_samples,
                eval_samples,
                baseline_epochs,
                bag_sizes: parse_usize_list(&bag_sizes)?,
                recovery_ratios: parse_f32_list(&recovery_ratios)?,
                lr,
                gate_tolerance,
                min_nominal_speedup,
                min_train_speedup,
                artifact_path: model_out,
            };
            let report = symliquid_metal::train_token_superposition_metal(
                Path::new("."),
                config,
                samples_per_launch,
            )?;
            let mut report_value = serde_json::to_value(&report)?;
            annotate_train_token_superposition_metal_report(
                &mut report_value,
                serde_json::json!({
                    "input": input,
                    "include_project_code": include_project_code,
                    "project_code_roots": project_code_roots,
                    "train_seed": train_seed,
                    "max_language_rows": max_language_rows,
                    "max_code_files": max_code_files,
                    "max_chars_per_doc": max_chars_per_doc,
                    "max_vocab": max_vocab,
                    "hv_dim": hv_dim,
                    "train_samples": train_samples,
                    "eval_samples": eval_samples,
                    "baseline_epochs": baseline_epochs,
                    "bag_sizes": bag_sizes,
                    "recovery_ratios": recovery_ratios,
                    "lr": lr,
                    "samples_per_launch": samples_per_launch,
                    "gate_tolerance": gate_tolerance,
                    "min_nominal_speedup": min_nominal_speedup,
                    "min_train_speedup": min_train_speedup,
                    "model_out": model_out_for_args,
                    "out": out
                }),
            );
            write_json_value(&out, &report_value)?;
            println!("{}", format_token_superposition_report(&report));
            println!("Wrote Metal token superposition report to {out}");
        }
        Command::TrainCodeRanker {
            candidate_manifest,
            trace_in,
            seed,
            holdout_ratio,
            epochs,
            hv_dim,
            lr,
            use_cuda_readout,
            model_out,
            candidate_out,
            training_examples_out,
            transfer_artifact_out,
            code_transfer_artifacts,
            out,
        } => {
            train_code_ranker(CodeRankerConfig {
                candidate_manifest,
                trace_in,
                seed,
                holdout_ratio,
                epochs,
                hv_dim,
                lr,
                use_cuda_readout,
                model_out,
                candidate_out,
                training_examples_out,
                transfer_artifact_out,
                code_transfer_artifacts,
                out,
            })?;
        }
        Command::TrainCodeTokenGenerator {
            task_manifest,
            training_sources,
            project_code_roots,
            seed,
            max_training_rows_per_source,
            max_project_files,
            max_candidates_per_task,
            checkpoint_out,
            out,
            report_out,
        } => {
            train_code_token_generator(CodeTokenGeneratorConfig {
                task_manifest,
                training_sources,
                project_code_roots,
                seed,
                max_training_rows_per_source,
                max_project_files,
                max_candidates_per_task,
                checkpoint_out,
                out,
                report_out,
            })?;
        }
        Command::TrainCodeLmClosure {
            private_curriculum,
            public_task_manifest,
            seed,
            hv_dim,
            max_vocab,
            epochs,
            lr,
            candidates_per_task,
            max_work_steps,
            use_cuda_readout,
            readout_eval_limit,
            aux_decoder_train_limit,
            checkpoint_only,
            checkpoint_out,
            checkpoint_in,
            private_candidate_out,
            public_candidate_out,
            report_out,
            sts_streams,
        } => {
            let config = CodeLmClosureConfig {
                private_curriculum,
                public_task_manifest,
                seed,
                hv_dim,
                max_vocab,
                epochs,
                lr,
                candidates_per_task,
                max_work_steps,
                use_cuda_readout,
                readout_eval_limit,
                aux_decoder_train_limit,
                checkpoint_only,
                checkpoint_out,
                checkpoint_in,
                private_candidate_out,
                public_candidate_out,
                report_out,
                sts_streams,
            };
            run_with_large_stack("code-lm-closure", move || train_code_lm_closure(config))?;
        }
        Command::GenerateCodeLmClosureFanout {
            private_curriculum,
            public_task_manifest,
            checkpoint_in,
            seed,
            candidates_per_task,
            private_candidate_out,
            public_candidate_out,
            report_out,
            sts_streams,
            transformer_hybrid_candidate_manifest,
            private_eval_limit,
            public_task_limit,
        } => {
            let config = CodeLmFanoutConfig {
                private_curriculum,
                public_task_manifest,
                checkpoint_in,
                seed,
                candidates_per_task,
                private_candidate_out,
                public_candidate_out,
                report_out,
                sts_streams,
                transformer_hybrid_candidate_manifest,
                private_eval_limit,
                public_task_limit,
            };
            run_with_large_stack("code-lm-fanout", move || {
                generate_code_lm_closure_fanout(config)
            })?;
        }
        Command::TrainStsParallelDecoder {
            input,
            seed,
            hv_dim,
            max_vocab,
            epochs,
            lr,
            max_generate_steps,
            max_train_rows,
            max_eval_rows,
            max_generate_rows,
            checkpoint_out,
            generation_out,
            report_out,
        } => {
            let config = StsParallelDecoderConfig {
                input,
                seed,
                hv_dim,
                max_vocab,
                epochs,
                lr,
                max_generate_steps,
                max_train_rows,
                max_eval_rows,
                max_generate_rows,
                checkpoint_out,
                generation_out,
                report_out,
            };
            run_with_large_stack("sts-parallel-decoder", move || {
                train_sts_parallel_decoder(config)
            })?;
        }
        Command::TrainBaseline {
            train_seed,
            eval_seed,
            cases_per_task,
            epochs,
            batch_size,
            hv_dim,
            lr,
            model_out,
            out,
        } => {
            let report = train_text_hash_baseline(StandaloneTrainConfig {
                train_seed,
                eval_seed,
                cases_per_task,
                epochs,
                batch_size,
                hv_dim,
                lr,
                symbolic_fallback: false,
                artifact_path: model_out,
            })?;
            write_standalone_train_report(&out, &report)?;
            println!(
                "{}",
                symliquid_core::benchmarks::format_standalone_train_report(&report)
            );
            println!("Wrote baseline training report to {out}");
        }
        Command::SeedSweep {
            train_seeds,
            eval_seed_base,
            cases_per_task,
            epochs,
            batch_size,
            hv_dim,
            lr,
            symbolic_fallback,
            out,
        } => {
            let seeds = parse_seed_list(&train_seeds)?;
            let report = run_seed_sweep(
                &seeds,
                eval_seed_base,
                StandaloneTrainConfig {
                    train_seed: 0,
                    eval_seed: eval_seed_base,
                    cases_per_task,
                    epochs,
                    batch_size,
                    hv_dim,
                    lr,
                    symbolic_fallback,
                    artifact_path: None,
                },
            )?;
            write_seed_sweep_report(&out, &report)?;
            println!("{}", format_seed_sweep_report(&report));
            println!("Wrote seed sweep report to {out}");
        }
        Command::BenchmarkBaseline {
            suite,
            baseline,
            seed,
            hv_dim,
            epochs,
            lr,
            out,
        } => {
            let suite = read_suite_json(&suite)?;
            let baseline = LocalBaselineKind::parse(&baseline)?;
            let report = run_local_baseline_suite(&suite, baseline, seed, hv_dim, epochs, lr)?;
            write_report(&out, &report)?;
            println!("{}", format_summary(&report.summary));
            println!("Wrote local baseline report to {out}");
        }
        Command::BabylmProbe {
            input,
            seed,
            limit,
            out_suite,
            out_report,
        } => {
            let suite = generate_babylm_probe_suite(&input, seed, limit)?;
            write_suite_json(&out_suite, &suite)?;
            let report = run_symliquid_suite(&suite, "symliquid-babylm-local-probe", false);
            write_report(&out_report, &report)?;
            println!(
                "Wrote BabyLM local probe suite '{}' with {} cases to {}",
                suite.name,
                suite.cases.len(),
                out_suite
            );
            println!("{}", format_summary(&report.summary));
            println!("Wrote BabyLM local probe report to {out_report}");
        }
        Command::TrainBabylmProbe {
            input,
            eval_input,
            train_seed,
            eval_seed,
            train_limit,
            eval_limit,
            steps,
            hv_dim,
            lr,
            stateful,
            pairwise_contrast,
            balance_rules,
            prior_weight,
            out,
        } => {
            let report = train_babylm_probe_scorer(BabyLmProbeTrainConfig {
                input_path: input,
                eval_input_path: eval_input,
                train_seed,
                eval_seed,
                train_limit,
                eval_limit,
                steps,
                hv_dim,
                lr,
                stateful,
                pairwise_contrast,
                balance_rules,
                prior_weight,
            })?;
            write_babylm_probe_train_report(&out, &report)?;
            println!("{}", format_babylm_probe_train_report(&report));
            println!("Wrote BabyLM probe training report to {out}");
        }
    }
    Ok(())
}

fn parse_run_mode(mode: &str) -> Result<RunMode, Box<dyn std::error::Error>> {
    match mode {
        "symliquid" => Ok(RunMode::SymLiquid),
        "local_baseline" => Ok(RunMode::LocalBaseline),
        "symliquid_augmented" => Ok(RunMode::SymLiquidAugmented),
        "manual" => Ok(RunMode::Manual),
        other => Err(format!(
            "unknown mode '{other}', expected symliquid/local_baseline/symliquid_augmented/manual"
        )
        .into()),
    }
}

fn parse_seed_list(seeds: &str) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    let parsed = seeds
        .split(',')
        .map(str::trim)
        .filter(|seed| !seed.is_empty())
        .map(str::parse::<u64>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if parsed.is_empty() {
        return Err("at least one seed is required".into());
    }
    Ok(parsed)
}

fn split_string_list(values: &str) -> Vec<String> {
    values
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn parse_usize_list(values: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let parsed = values
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if parsed.is_empty() {
        return Err("at least one integer value is required".into());
    }
    Ok(parsed)
}

fn parse_f32_list(values: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let parsed = values
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::parse::<f32>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if parsed.is_empty() {
        return Err("at least one float value is required".into());
    }
    if parsed
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("float grid values must be finite and non-negative".into());
    }
    Ok(parsed)
}

#[cfg(feature = "cuda")]
fn build_rollout_cuda_sweep_report(
    build: RolloutCudaSweepBuild,
) -> Result<RolloutCudaSweepReport, Box<dyn std::error::Error>> {
    if build.runs.is_empty() {
        return Err("rollout sweep produced no runs".into());
    }
    let accuracies = build
        .runs
        .iter()
        .map(|run| run.eval.summary.accuracy)
        .collect::<Vec<_>>();
    let residuals = build
        .runs
        .iter()
        .map(|run| run.eval.summary.residual)
        .collect::<Vec<_>>();
    let (mean_accuracy, std_accuracy) = mean_std(&accuracies);
    let (mean_residual, std_residual) = mean_std(&residuals);
    let mut best_index = 0usize;
    for idx in 1..build.runs.len() {
        let best = &build.runs[best_index].eval.summary;
        let candidate = &build.runs[idx].eval.summary;
        let tied_accuracy = (candidate.accuracy - best.accuracy).abs() <= f32::EPSILON;
        if candidate.accuracy > best.accuracy
            || (tied_accuracy && candidate.residual < best.residual)
        {
            best_index = idx;
        }
    }
    let accepted_state_candidates = build
        .runs
        .iter()
        .filter(|run| {
            run.state_training
                .as_ref()
                .map(|state| state.accepted)
                .unwrap_or(false)
        })
        .count();
    let best_accuracy = build.runs[best_index].eval.summary.accuracy;
    let best_residual = build.runs[best_index].eval.summary.residual;
    Ok(RolloutCudaSweepReport {
        train_seeds: build.train_seeds,
        eval_seed_base: build.eval_seed_base,
        cases_per_task: build.cases_per_task,
        epochs: build.epochs,
        state_epoch_grid: build.state_epoch_grid,
        state_lr_grid: build.state_lr_grid,
        samples_per_launch: build.samples_per_launch,
        probe_cases_per_task: build.probe_cases_per_task,
        rollout_batch: build.rollout_batch,
        obs_dim: build.obs_dim,
        hidden_dim: build.hidden_dim,
        reservoir_dim: build.reservoir_dim,
        hv_dim: build.hv_dim,
        seq_len: build.seq_len,
        lr: build.lr,
        runs: build.runs,
        best_index,
        best_accuracy,
        best_residual,
        mean_accuracy,
        std_accuracy,
        mean_residual,
        std_residual,
        accepted_state_candidates,
    })
}

#[cfg(feature = "cuda")]
fn write_rollout_cuda_sweep_report(
    path: &str,
    report: &RolloutCudaSweepReport,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn format_rollout_cuda_sweep_report(report: &RolloutCudaSweepReport) -> String {
    let mut out = String::new();
    out.push_str("CUDA rollout sweep\n");
    out.push_str(&format!("Train seeds: {:?}\n", report.train_seeds));
    out.push_str(&format!("Eval seed base: {}\n", report.eval_seed_base));
    out.push_str(&format!("Cases per task: {}\n", report.cases_per_task));
    out.push_str(&format!(
        "Probe cases per task: {}\n",
        report.probe_cases_per_task
    ));
    out.push_str(&format!("Runs: {}\n", report.runs.len()));
    out.push_str(&format!(
        "Accuracy: mean={:.3} std={:.3} best={:.3}\n",
        report.mean_accuracy, report.std_accuracy, report.best_accuracy
    ));
    out.push_str(&format!(
        "Residual: mean={:.3} std={:.3} best={:.3}\n",
        report.mean_residual, report.std_residual, report.best_residual
    ));
    out.push_str(&format!(
        "Accepted state candidates: {}\n",
        report.accepted_state_candidates
    ));
    let best = &report.runs[report.best_index];
    let state = best.state_training.as_ref();
    out.push_str(&format!(
        "Best run: index={} train_seed={} eval_seed={} feature_set={} state_epochs={} state_lr={:.5}\n",
        report.best_index,
        best.train_seed,
        best.eval_seed,
        best.feature_set,
        state.map(|entry| entry.state_epochs).unwrap_or(0),
        state.map(|entry| entry.state_lr).unwrap_or(0.0)
    ));
    out
}

#[cfg(feature = "cuda")]
fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f32>()
        / values.len() as f32;
    (mean, variance.sqrt())
}
