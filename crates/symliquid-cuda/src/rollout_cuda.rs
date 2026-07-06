#[cfg(feature = "cuda")]
use std::collections::{BTreeMap, BTreeSet};

use symliquid_core::error::{Result, SymError};
use symliquid_core::tensor::Tensor;

#[cfg(feature = "cuda")]
mod feature_support;
#[cfg(feature = "cuda")]
use feature_support::*;

#[derive(Debug, Clone)]
pub struct RolloutConfig {
    pub batch: usize,
    pub obs_dim: usize,
    pub hidden_dim: usize,
    pub reservoir_dim: usize,
    pub hv_dim: usize,
    pub dt: f32,
    pub alpha: f32,
    pub memory_decay: f32,
}

#[derive(Debug, Clone)]
pub struct RolloutParams {
    pub obs_to_h: Vec<f32>,
    pub h_recurrent: Vec<f32>,
    pub h_bias: Vec<f32>,
    pub reservoir_input: Vec<f32>,
    pub reservoir_recurrent: Vec<f32>,
    pub reservoir_bias: Vec<f32>,
    pub hv_proj: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct RolloutState {
    pub h: Tensor,
    pub r: Tensor,
    pub memory: Tensor,
}

#[cfg(feature = "cuda")]
const ROLLOUT_KERNELS: &str = include_str!("../kernels/rollout_kernels.cu");

#[cfg(feature = "cuda")]
struct RolloutKernelCache {
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    function: cudarc::driver::CudaFunction,
}

#[cfg(feature = "cuda")]
static ROLLOUT_KERNEL_CACHE: std::sync::OnceLock<std::result::Result<RolloutKernelCache, String>> =
    std::sync::OnceLock::new();

#[cfg(feature = "cuda")]
fn rollout_kernel_cache() -> Result<&'static RolloutKernelCache> {
    match ROLLOUT_KERNEL_CACHE.get_or_init(init_rollout_kernel_cache) {
        Ok(cache) => Ok(cache),
        Err(error) => Err(SymError::InvalidArgument(format!(
            "CUDA rollout kernel cache initialization failed: {error}"
        ))),
    }
}

#[cfg(feature = "cuda")]
fn init_rollout_kernel_cache() -> std::result::Result<RolloutKernelCache, String> {
    use cudarc::driver::CudaContext;
    use cudarc::nvrtc::compile_ptx;

    let ctx = CudaContext::new(0).map_err(|error| error.to_string())?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(ROLLOUT_KERNELS).map_err(|error| error.to_string())?;
    let module = ctx.load_module(ptx).map_err(|error| error.to_string())?;
    let function = module
        .load_function("rollout_state_update_kernel")
        .map_err(|error| error.to_string())?;
    Ok(RolloutKernelCache { stream, function })
}

pub fn rollout_sequence_cpu(
    observations: &Tensor,
    state: &RolloutState,
    params: &RolloutParams,
    config: &RolloutConfig,
) -> Result<RolloutState> {
    validate_rollout(observations, state, params, config)?;
    let steps = observations.rows / config.batch;
    let mut h_prev = state.h.clone();
    let mut r_prev = state.r.clone();
    let mut memory_prev = state.memory.clone();

    for step in 0..steps {
        let mut h_next = Tensor::zeros(config.batch, config.hidden_dim);
        let mut r_next = Tensor::zeros(config.batch, config.reservoir_dim);
        let mut memory_next = Tensor::zeros(config.batch, config.hv_dim);

        for b in 0..config.batch {
            let obs_row = step * config.batch + b;
            for j in 0..config.hidden_dim {
                let mut pre = params.h_bias[j];
                for i in 0..config.obs_dim {
                    pre += params.obs_to_h[j * config.obs_dim + i] * observations.get(obs_row, i);
                }
                for k in 0..config.hidden_dim {
                    pre += params.h_recurrent[j * config.hidden_dim + k] * h_prev.get(b, k);
                }
                let candidate = pre.tanh();
                let tau = symliquid_core::tensor::softplus(pre) + 1.0e-3;
                let dh = (-h_prev.get(b, j) + candidate) / tau;
                h_next.set(b, j, h_prev.get(b, j) + config.dt * dh);
            }

            for j in 0..config.reservoir_dim {
                let mut pre = params.reservoir_bias[j];
                for i in 0..config.hidden_dim {
                    pre += params.reservoir_input[j * config.hidden_dim + i] * h_next.get(b, i);
                }
                for k in 0..config.reservoir_dim {
                    pre +=
                        params.reservoir_recurrent[j * config.reservoir_dim + k] * r_prev.get(b, k);
                }
                let value = (1.0 - config.alpha) * r_prev.get(b, j) + config.alpha * pre.tanh();
                r_next.set(b, j, value);
            }

            for j in 0..config.hv_dim {
                let mut dot = 0.0;
                for k in 0..config.reservoir_dim {
                    dot += params.hv_proj[j * config.reservoir_dim + k] * r_next.get(b, k);
                }
                let hv = if dot >= 0.0 { 1.0 } else { -1.0 };
                memory_next.set(b, j, config.memory_decay * memory_prev.get(b, j) + hv);
            }
        }

        h_prev = h_next;
        r_prev = r_next;
        memory_prev = memory_next;
    }

    Ok(RolloutState {
        h: h_prev,
        r: r_prev,
        memory: memory_prev,
    })
}

#[cfg(feature = "cuda")]
pub fn rollout_sequence_cuda(
    observations: &Tensor,
    state: &RolloutState,
    params: &RolloutParams,
    config: &RolloutConfig,
) -> Result<RolloutState> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    validate_rollout(observations, state, params, config)?;
    let steps = observations.rows / config.batch;
    let kernel = rollout_kernel_cache()?;
    let stream = &kernel.stream;
    let function = &kernel.function;

    let observations_dev = stream.clone_htod(&observations.data).map_err(cuda_error)?;
    let mut h_current = stream.clone_htod(&state.h.data).map_err(cuda_error)?;
    let mut h_next = stream
        .clone_htod(&vec![0.0f32; state.h.data.len()])
        .map_err(cuda_error)?;
    let mut r_current = stream.clone_htod(&state.r.data).map_err(cuda_error)?;
    let mut r_next = stream
        .clone_htod(&vec![0.0f32; state.r.data.len()])
        .map_err(cuda_error)?;
    let mut memory_current = stream.clone_htod(&state.memory.data).map_err(cuda_error)?;
    let mut memory_next = stream
        .clone_htod(&vec![0.0f32; state.memory.data.len()])
        .map_err(cuda_error)?;

    let obs_to_h_dev = stream.clone_htod(&params.obs_to_h).map_err(cuda_error)?;
    let h_recurrent_dev = stream.clone_htod(&params.h_recurrent).map_err(cuda_error)?;
    let h_bias_dev = stream.clone_htod(&params.h_bias).map_err(cuda_error)?;
    let reservoir_input_dev = stream
        .clone_htod(&params.reservoir_input)
        .map_err(cuda_error)?;
    let reservoir_recurrent_dev = stream
        .clone_htod(&params.reservoir_recurrent)
        .map_err(cuda_error)?;
    let reservoir_bias_dev = stream
        .clone_htod(&params.reservoir_bias)
        .map_err(cuda_error)?;
    let hv_proj_dev = stream.clone_htod(&params.hv_proj).map_err(cuda_error)?;

    let cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (config.batch as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let obs_dim = config.obs_dim as i32;
    let hidden_dim = config.hidden_dim as i32;
    let reservoir_dim = config.reservoir_dim as i32;
    let hv_dim = config.hv_dim as i32;

    for step in 0..steps {
        let observation_offset = (step * config.batch * config.obs_dim) as i32;
        let mut args = stream.launch_builder(function);
        args.arg(&observations_dev);
        args.arg(&observation_offset);
        args.arg(&h_current);
        args.arg(&r_current);
        args.arg(&memory_current);
        args.arg(&mut h_next);
        args.arg(&mut r_next);
        args.arg(&mut memory_next);
        args.arg(&obs_to_h_dev);
        args.arg(&h_recurrent_dev);
        args.arg(&h_bias_dev);
        args.arg(&reservoir_input_dev);
        args.arg(&reservoir_recurrent_dev);
        args.arg(&reservoir_bias_dev);
        args.arg(&hv_proj_dev);
        args.arg(&obs_dim);
        args.arg(&hidden_dim);
        args.arg(&reservoir_dim);
        args.arg(&hv_dim);
        args.arg(&config.dt);
        args.arg(&config.alpha);
        args.arg(&config.memory_decay);
        unsafe {
            args.launch(cfg).map_err(cuda_error)?;
        }
        std::mem::swap(&mut h_current, &mut h_next);
        std::mem::swap(&mut r_current, &mut r_next);
        std::mem::swap(&mut memory_current, &mut memory_next);
    }

    let h = Tensor::new(
        config.batch,
        config.hidden_dim,
        stream.clone_dtoh(&h_current).map_err(cuda_error)?,
    )?;
    let r = Tensor::new(
        config.batch,
        config.reservoir_dim,
        stream.clone_dtoh(&r_current).map_err(cuda_error)?,
    )?;
    let memory = Tensor::new(
        config.batch,
        config.hv_dim,
        stream.clone_dtoh(&memory_current).map_err(cuda_error)?,
    )?;
    Ok(RolloutState { h, r, memory })
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct RolloutFeatureConfig {
    pub obs_dim: usize,
    pub hidden_dim: usize,
    pub reservoir_dim: usize,
    pub hv_dim: usize,
    pub seq_len: usize,
    pub rollout_batch: usize,
    pub dt: f32,
    pub alpha: f32,
    pub memory_decay: f32,
}

#[cfg(feature = "cuda")]
impl Default for RolloutFeatureConfig {
    fn default() -> Self {
        Self {
            obs_dim: 64,
            hidden_dim: 96,
            reservoir_dim: 128,
            hv_dim: 2048,
            seq_len: 64,
            rollout_batch: 32,
            dt: 0.1,
            alpha: 0.25,
            memory_decay: 0.98,
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct RolloutProbeConfig {
    epochs: usize,
    lr: f32,
    samples_per_launch: usize,
    seed: u64,
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
struct RolloutProbeTrace {
    loss: f32,
    accuracy: f32,
    state_alignment: f32,
    cases: usize,
    task_breakdown: BTreeMap<String, symliquid_core::benchmarks::TaskBreakdown>,
}

#[cfg(feature = "cuda")]
struct StateBlockUpdate<'a> {
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

#[cfg(feature = "cuda")]
struct StateDynamicsSample<'a> {
    obs_features: &'a [f32],
    hidden_state: &'a [f32],
    reservoir_state: &'a [f32],
    hidden_target: &'a [f32],
    reservoir_target: &'a [f32],
}

#[cfg(feature = "cuda")]
struct TaskGatedProbeInputs<'a> {
    fit_cases: &'a [symliquid_core::benchmarks::BenchmarkCase],
    validation_cases: &'a [symliquid_core::benchmarks::BenchmarkCase],
    labels: &'a [String],
    base_params: &'a RolloutParams,
    candidate_params: &'a RolloutParams,
    candidate_tasks: &'a BTreeSet<String>,
    config: &'a RolloutFeatureConfig,
    probe: RolloutProbeConfig,
}

#[cfg(feature = "cuda")]
struct TaskHeadReadouts {
    heads: BTreeMap<String, symliquid_core::train::LinearReadout>,
    loss: f32,
    accuracy: f32,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct TaskResidualAdapter {
    input_dim: usize,
    output_dim: usize,
    rank: usize,
    projection: Vec<f32>,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

#[cfg(feature = "cuda")]
struct TaskResidualAdapters {
    adapters: BTreeMap<String, TaskResidualAdapter>,
    rank: usize,
    loss: f32,
    accuracy: f32,
}

#[cfg(feature = "cuda")]
struct TaskHeadTrainInputs<'a> {
    cases: &'a [symliquid_core::benchmarks::BenchmarkCase],
    features: &'a Tensor,
    targets: &'a [usize],
    labels: &'a [String],
    hv_dim: usize,
    epochs: usize,
    lr: f32,
    samples_per_launch: usize,
    seed: u64,
}

#[cfg(feature = "cuda")]
struct ResidualAdapterTrainInputs<'a> {
    cases: &'a [symliquid_core::benchmarks::BenchmarkCase],
    features: &'a Tensor,
    targets: &'a [usize],
    labels: &'a [String],
    shared_readout: &'a symliquid_core::train::LinearReadout,
    hv_dim: usize,
    epochs: usize,
    lr: f32,
    rank: usize,
    seed: u64,
}

#[cfg(feature = "cuda")]
struct ReadoutProbeTrace {
    accuracy: f32,
    cases: usize,
    task_breakdown: BTreeMap<String, symliquid_core::benchmarks::TaskBreakdown>,
}

#[cfg(feature = "cuda")]
pub fn train_standalone_rollout_cuda(
    config: symliquid_core::benchmarks::StandaloneTrainConfig,
    rollout_config: RolloutFeatureConfig,
    samples_per_launch: usize,
    state_epochs: usize,
    state_lr: f32,
    probe_cases_per_task: usize,
) -> Result<symliquid_core::benchmarks::StandaloneTrainReport> {
    use std::time::Instant;

    use rand::SeedableRng;
    use symliquid_core::benchmarks::{
        generate_cgs_hard_suite, standalone_output_vocab, ReadoutRoutingDiagnostics,
        StandaloneTrainReport, StateTrainingDiagnostics,
    };
    use symliquid_core::train::LinearReadout;

    let train_start = Instant::now();
    let mut timing_breakdown_ms = BTreeMap::new();
    let setup_start = Instant::now();
    let train_suite = generate_cgs_hard_suite(config.train_seed, config.cases_per_task);
    let eval_suite = generate_cgs_hard_suite(config.eval_seed, config.cases_per_task);
    let probe_cases_per_task = if probe_cases_per_task == 0 {
        config.cases_per_task.max(50)
    } else {
        probe_cases_per_task.max(1)
    };
    let mut probe_cases =
        generate_cgs_hard_suite(config.train_seed ^ 0x51A7_EC05, probe_cases_per_task).cases;
    let mut probe_cases_b =
        generate_cgs_hard_suite(config.eval_seed ^ 0xC0DE_5EED, probe_cases_per_task).cases;
    probe_cases.append(&mut probe_cases_b);
    let labels = standalone_output_vocab();
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.train_seed ^ 0xC10C_10C0);
    let mut params = init_rollout_params(&rollout_config, &mut rng);
    let mut feature_set =
        "cuda_liquid_reservoir_vsa_rollout_entity_slot_memory_frozen_projection".to_string();
    let mut state_training = None;
    let mut task_gated_params: Option<(RolloutParams, RolloutParams, BTreeSet<String>)> = None;
    let mut shared_readout = LinearReadout::new(rollout_config.hv_dim, labels.len(), &mut rng);
    timing_breakdown_ms.insert("setup".to_string(), setup_start.elapsed().as_millis());

    if state_epochs > 0 && state_lr > 0.0 {
        let state_training_start = Instant::now();
        let base_params = params.clone();
        let mut candidate_params = params.clone();
        train_state_params_from_rollouts(
            &train_suite.cases,
            &labels,
            &mut candidate_params,
            &rollout_config,
            state_epochs,
            state_lr,
        )?;
        let probe_epochs = config.epochs.clamp(1, 3);
        let base_probe = probe_rollout_params(
            &train_suite.cases,
            &probe_cases,
            &labels,
            &base_params,
            &rollout_config,
            RolloutProbeConfig {
                epochs: probe_epochs,
                lr: config.lr,
                samples_per_launch,
                seed: config.train_seed ^ 0xBADA_551E,
            },
        )?;
        let candidate_probe = probe_rollout_params(
            &train_suite.cases,
            &probe_cases,
            &labels,
            &candidate_params,
            &rollout_config,
            RolloutProbeConfig {
                epochs: probe_epochs,
                lr: config.lr,
                samples_per_launch,
                seed: config.train_seed ^ 0xBADA_551E,
            },
        )?;
        let worst_candidate_task_accuracy_delta = worst_probe_task_accuracy_delta(
            &base_probe.task_breakdown,
            &candidate_probe.task_breakdown,
        );
        let candidate_state_tasks =
            candidate_state_task_gate(&base_probe.task_breakdown, &candidate_probe.task_breakdown);
        let accepted = candidate_probe.accuracy >= base_probe.accuracy + 0.02
            && candidate_probe.loss <= base_probe.loss * 1.05
            && worst_candidate_task_accuracy_delta >= -0.05;
        let gated_probe = if !accepted
            && candidate_probe.accuracy >= base_probe.accuracy
            && !candidate_state_tasks.is_empty()
        {
            Some(probe_rollout_task_gated_params(TaskGatedProbeInputs {
                fit_cases: &train_suite.cases,
                validation_cases: &probe_cases,
                labels: &labels,
                base_params: &base_params,
                candidate_params: &candidate_params,
                candidate_tasks: &candidate_state_tasks,
                config: &rollout_config,
                probe: RolloutProbeConfig {
                    epochs: probe_epochs,
                    lr: config.lr,
                    samples_per_launch,
                    seed: config.train_seed ^ 0xA11E_6A7E,
                },
            })?)
        } else {
            None
        };
        let gated_worst_task_accuracy_delta = gated_probe
            .as_ref()
            .map(|probe| {
                worst_probe_task_accuracy_delta(&base_probe.task_breakdown, &probe.task_breakdown)
            })
            .unwrap_or(0.0);
        let task_gated_candidate = gated_probe
            .as_ref()
            .map(|probe| {
                probe.accuracy >= base_probe.accuracy + 0.02
                    && probe.loss <= base_probe.loss * 1.05
                    && gated_worst_task_accuracy_delta >= -0.05
            })
            .unwrap_or(false);
        state_training = Some(StateTrainingDiagnostics {
            attempted: true,
            accepted,
            state_epochs,
            state_lr,
            probe_cases: base_probe.cases,
            probe_accuracy_metric: "masked_allowed_action_or_answer_candidate_accuracy".to_string(),
            base_probe_loss: base_probe.loss,
            base_probe_accuracy: base_probe.accuracy,
            base_state_alignment: base_probe.state_alignment,
            base_probe_task_breakdown: base_probe.task_breakdown.clone(),
            candidate_probe_loss: candidate_probe.loss,
            candidate_probe_accuracy: candidate_probe.accuracy,
            candidate_state_alignment: candidate_probe.state_alignment,
            candidate_probe_task_breakdown: candidate_probe.task_breakdown.clone(),
            worst_probe_task_accuracy_delta: worst_candidate_task_accuracy_delta,
            task_gated_candidate,
            candidate_state_tasks: candidate_state_tasks.iter().cloned().collect(),
            gated_probe_loss: gated_probe.as_ref().map(|probe| probe.loss).unwrap_or(0.0),
            gated_probe_accuracy: gated_probe
                .as_ref()
                .map(|probe| probe.accuracy)
                .unwrap_or(0.0),
            gated_worst_task_accuracy_delta,
            acceptance_rule:
                "candidate_masked_accuracy >= base_masked_accuracy + 0.02 && candidate_loss <= base_loss * 1.05 && worst_task_accuracy_delta >= -0.05; task-gated candidates must pass the same rule after routing by probe-positive task families"
                    .to_string(),
        });
        if accepted {
            params = candidate_params;
            feature_set =
                "cuda_liquid_reservoir_vsa_rollout_entity_slot_memory_trainable_state".to_string();
        } else if task_gated_candidate {
            task_gated_params = Some((base_params, candidate_params, candidate_state_tasks));
            feature_set =
                "cuda_liquid_reservoir_vsa_rollout_entity_slot_memory_task_gated_state_candidate"
                    .to_string();
        } else {
            params = base_params;
            feature_set =
                "cuda_liquid_reservoir_vsa_rollout_entity_slot_memory_frozen_state_rejected_trainable_candidate"
                    .to_string();
        }
        timing_breakdown_ms.insert(
            "state_training_and_probe".to_string(),
            state_training_start.elapsed().as_millis(),
        );
    }

    let train_feature_start = Instant::now();
    let (train_cases, train_features, train_targets) =
        if let Some((base_params, candidate_params, tasks)) = &task_gated_params {
            rollout_labeled_cases_features_task_gated_cuda(
                &train_suite.cases,
                &labels,
                base_params,
                candidate_params,
                tasks,
                &rollout_config,
            )?
        } else {
            rollout_labeled_cases_features_cuda(
                &train_suite.cases,
                &labels,
                &params,
                &rollout_config,
            )?
        };
    timing_breakdown_ms.insert(
        "train_feature_rollout".to_string(),
        train_feature_start.elapsed().as_millis(),
    );
    let curriculum_start = Instant::now();
    let (train_cases, train_features, train_targets) = retrieval_first_curriculum(
        train_cases,
        train_features,
        train_targets,
        rollout_config.hv_dim,
    )?;
    timing_breakdown_ms.insert(
        "retrieval_curriculum".to_string(),
        curriculum_start.elapsed().as_millis(),
    );
    feature_set = format!("{feature_set}_retrieval_curriculum");
    let task_head_start = Instant::now();
    let task_heads = train_task_head_readouts_cuda(TaskHeadTrainInputs {
        cases: &train_cases,
        features: &train_features,
        targets: &train_targets,
        labels: &labels,
        hv_dim: rollout_config.hv_dim,
        epochs: config.epochs,
        lr: config.lr,
        samples_per_launch,
        seed: config.train_seed ^ 0x7A5C_2EAD,
    })?;
    timing_breakdown_ms.insert(
        "task_head_cuda_readouts".to_string(),
        task_head_start.elapsed().as_millis(),
    );
    let shared_readout_start = Instant::now();
    let shared_trace = crate::readout_cuda::train_readout_sgd_cuda(
        &train_features,
        &train_targets,
        &mut shared_readout,
        config.epochs,
        config.lr,
        samples_per_launch,
    )?;
    timing_breakdown_ms.insert(
        "shared_cuda_readout".to_string(),
        shared_readout_start.elapsed().as_millis(),
    );
    let residual_adapter_rank = labels.len().clamp(16, 64);
    let residual_adapter_start = Instant::now();
    let residual_adapters = train_task_residual_adapters(ResidualAdapterTrainInputs {
        cases: &train_cases,
        features: &train_features,
        targets: &train_targets,
        labels: &labels,
        shared_readout: &shared_readout,
        hv_dim: rollout_config.hv_dim,
        epochs: config.epochs.saturating_mul(3).max(1),
        lr: config.lr,
        rank: residual_adapter_rank,
        seed: config.train_seed ^ 0xA9A9_5EED,
    })?;
    timing_breakdown_ms.insert(
        "residual_adapters".to_string(),
        residual_adapter_start.elapsed().as_millis(),
    );
    let train_runtime_ms = train_start.elapsed().as_millis();
    let trained_examples = train_targets.len() * config.epochs;
    let train_examples_per_second =
        trained_examples as f32 / (train_runtime_ms.max(1) as f32 / 1000.0);

    let eval_feature_start = Instant::now();
    let eval_features = if let Some((base_params, candidate_params, tasks)) = &task_gated_params {
        rollout_features_task_gated_cuda(
            &eval_suite.cases,
            base_params,
            candidate_params,
            tasks,
            &rollout_config,
        )?
    } else {
        rollout_features_cuda(&eval_suite.cases, &params, &rollout_config)?
    };
    timing_breakdown_ms.insert(
        "eval_feature_rollout".to_string(),
        eval_feature_start.elapsed().as_millis(),
    );
    let readout_probe_start = Instant::now();
    let readout_probe_features =
        if let Some((base_params, candidate_params, tasks)) = &task_gated_params {
            rollout_features_task_gated_cuda(
                &probe_cases,
                base_params,
                candidate_params,
                tasks,
                &rollout_config,
            )?
        } else {
            rollout_features_cuda(&probe_cases, &params, &rollout_config)?
        };
    let shared_probe = probe_linear_readout(
        &probe_cases,
        &readout_probe_features,
        &labels,
        &shared_readout,
    )?;
    let task_head_probe = probe_task_head_readouts(
        &probe_cases,
        &readout_probe_features,
        &labels,
        &task_heads.heads,
    )?;
    let residual_adapter_probe = probe_residual_adapter_readouts(
        &probe_cases,
        &readout_probe_features,
        &labels,
        &shared_readout,
        &residual_adapters.adapters,
    )?;
    let residual_adapter_worst_task_accuracy_delta = worst_probe_task_accuracy_delta(
        &shared_probe.task_breakdown,
        &residual_adapter_probe.task_breakdown,
    );
    let accepted_residual_adapters = residual_adapter_probe.accuracy
        >= shared_probe.accuracy + 0.02
        && residual_adapter_worst_task_accuracy_delta >= -0.05;
    let task_head_worst_task_accuracy_delta = worst_probe_task_accuracy_delta(
        &shared_probe.task_breakdown,
        &task_head_probe.task_breakdown,
    );
    let accepted_task_heads = task_head_probe.accuracy >= shared_probe.accuracy + 0.02
        && task_head_worst_task_accuracy_delta >= -0.05;
    let selected_readout = if accepted_residual_adapters {
        "residual_adapters"
    } else if accepted_task_heads {
        "task_heads"
    } else {
        "shared_head"
    };
    let readout_routing = Some(ReadoutRoutingDiagnostics {
        selected_readout: selected_readout.to_string(),
        probe_cases: shared_probe.cases,
        shared_probe_accuracy: shared_probe.accuracy,
        residual_adapter_probe_accuracy: residual_adapter_probe.accuracy,
        residual_adapter_worst_task_accuracy_delta,
        accepted_residual_adapters,
        residual_adapter_rank: residual_adapters.rank,
        task_head_probe_accuracy: task_head_probe.accuracy,
        task_head_worst_task_accuracy_delta,
        accepted_task_heads,
        acceptance_rule:
            "residual_adapter_probe_accuracy >= shared_probe_accuracy + 0.02 && residual_adapter_worst_task_accuracy_delta >= -0.05; full task heads use the same guarded rule as a diagnostic backup"
                .to_string(),
    });
    timing_breakdown_ms.insert(
        "readout_probe_and_routing".to_string(),
        readout_probe_start.elapsed().as_millis(),
    );
    let (model_id, eval, train_loss, train_accuracy, feature_set) = if accepted_residual_adapters {
        (
            "symliquid-standalone-cuda-rollout-residual-adapters",
            evaluate_residual_adapter_readouts(
                &eval_suite,
                &eval_features,
                &labels,
                &shared_readout,
                &residual_adapters.adapters,
                "symliquid-standalone-cuda-rollout-residual-adapters",
            )?,
            residual_adapters.loss,
            residual_adapters.accuracy,
            format!("{feature_set}_residual_adapters"),
        )
    } else if accepted_task_heads {
        (
            "symliquid-standalone-cuda-rollout-task-heads",
            evaluate_task_head_readouts(
                &eval_suite,
                &eval_features,
                &labels,
                &task_heads.heads,
                "symliquid-standalone-cuda-rollout-task-heads",
            )?,
            task_heads.loss,
            task_heads.accuracy,
            format!("{feature_set}_task_heads"),
        )
    } else {
        (
            "symliquid-standalone-cuda-rollout-readout",
            evaluate_shared_readout(
                &eval_suite,
                &eval_features,
                &labels,
                &shared_readout,
                "symliquid-standalone-cuda-rollout-readout",
            )?,
            shared_trace.loss,
            shared_trace.accuracy,
            format!("{feature_set}_shared_head"),
        )
    };

    if let Some(path) = &config.artifact_path {
        eprintln!(
            "Skipping artifact write for rollout readout routing; multi-head artifact format is not implemented yet: {path}"
        );
    }

    timing_breakdown_ms.insert("total".to_string(), train_start.elapsed().as_millis());
    let rollout_batch = rollout_config.rollout_batch.max(1);
    let train_rollout_launches =
        train_suite.cases.len().div_ceil(rollout_batch) * rollout_config.seq_len;
    let eval_rollout_launches =
        eval_suite.cases.len().div_ceil(rollout_batch) * rollout_config.seq_len;
    let probe_rollout_launches = probe_cases.len().div_ceil(rollout_batch) * rollout_config.seq_len;
    let state_probe_multiplier = if state_epochs > 0 && state_lr > 0.0 {
        2 + usize::from(task_gated_params.is_some())
    } else {
        0
    };
    let readout_chunk = samples_per_launch.max(1);
    let shared_readout_launches = config.epochs * train_features.rows.div_ceil(readout_chunk);
    let task_head_launches =
        config.epochs * train_features.rows.div_ceil(readout_chunk) * task_heads.heads.len().max(1);
    let kernel_launches = train_rollout_launches
        + eval_rollout_launches
        + probe_rollout_launches
        + probe_rollout_launches * state_probe_multiplier
        + shared_readout_launches
        + task_head_launches;

    Ok(StandaloneTrainReport {
        model_id: model_id.to_string(),
        feature_set,
        train_seed: config.train_seed,
        eval_seed: config.eval_seed,
        cases_per_task: config.cases_per_task,
        epochs: config.epochs,
        batch_size: 1,
        hv_dim: rollout_config.hv_dim,
        labels: labels.len(),
        symbolic_fallback: false,
        train_runtime_ms,
        train_examples_per_second,
        train_loss,
        train_accuracy,
        readout_routing,
        state_training,
        runtime_profile: crate::device::runtime_profile(true),
        timing_breakdown_ms,
        kernel_launches,
        cuda_fallback: false,
        eval,
    })
}

#[cfg(feature = "cuda")]
fn rollout_labeled_features_cuda(
    cases: &[symliquid_core::benchmarks::BenchmarkCase],
    labels: &[String],
    params: &RolloutParams,
    config: &RolloutFeatureConfig,
) -> Result<(Tensor, Vec<usize>)> {
    let (_, features, targets) =
        rollout_labeled_cases_features_cuda(cases, labels, params, config)?;
    Ok((features, targets))
}

#[cfg(feature = "cuda")]
fn rollout_labeled_cases_features_cuda(
    cases: &[symliquid_core::benchmarks::BenchmarkCase],
    labels: &[String],
    params: &RolloutParams,
    config: &RolloutFeatureConfig,
) -> Result<(
    Vec<symliquid_core::benchmarks::BenchmarkCase>,
    Tensor,
    Vec<usize>,
)> {
    let (selected, targets) = labeled_cases_and_targets(cases, labels);
    let features = rollout_features_cuda(&selected, params, config)?;
    Ok((selected, features, targets))
}

#[cfg(feature = "cuda")]
fn rollout_labeled_features_task_gated_cuda(
    cases: &[symliquid_core::benchmarks::BenchmarkCase],
    labels: &[String],
    base_params: &RolloutParams,
    candidate_params: &RolloutParams,
    candidate_tasks: &BTreeSet<String>,
    config: &RolloutFeatureConfig,
) -> Result<(Tensor, Vec<usize>)> {
    let (_, features, targets) = rollout_labeled_cases_features_task_gated_cuda(
        cases,
        labels,
        base_params,
        candidate_params,
        candidate_tasks,
        config,
    )?;
    Ok((features, targets))
}

#[cfg(feature = "cuda")]
fn rollout_labeled_cases_features_task_gated_cuda(
    cases: &[symliquid_core::benchmarks::BenchmarkCase],
    labels: &[String],
    base_params: &RolloutParams,
    candidate_params: &RolloutParams,
    candidate_tasks: &BTreeSet<String>,
    config: &RolloutFeatureConfig,
) -> Result<(
    Vec<symliquid_core::benchmarks::BenchmarkCase>,
    Tensor,
    Vec<usize>,
)> {
    let (selected, targets) = labeled_cases_and_targets(cases, labels);
    let features = rollout_features_task_gated_cuda(
        &selected,
        base_params,
        candidate_params,
        candidate_tasks,
        config,
    )?;
    Ok((selected, features, targets))
}

#[cfg(feature = "cuda")]
fn train_task_head_readouts_cuda(inputs: TaskHeadTrainInputs<'_>) -> Result<TaskHeadReadouts> {
    use rand::SeedableRng;
    use symliquid_core::train::LinearReadout;

    inputs
        .features
        .ensure_cols(inputs.hv_dim, "task-head features")?;
    if inputs.cases.len() != inputs.features.rows || inputs.targets.len() != inputs.features.rows {
        return Err(SymError::Shape(format!(
            "task-head rows mismatch: cases={} features={} targets={}",
            inputs.cases.len(),
            inputs.features.rows,
            inputs.targets.len()
        )));
    }

    let mut rows_by_task: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (row, case) in inputs.cases.iter().enumerate() {
        rows_by_task.entry(case.task.clone()).or_default().push(row);
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(inputs.seed);
    let mut heads = BTreeMap::new();
    let mut weighted_loss = 0.0;
    let mut weighted_accuracy = 0.0;
    let mut total_rows = 0usize;

    for (task, rows) in rows_by_task {
        let task_features = select_feature_rows(inputs.features, &rows, inputs.hv_dim)?;
        let task_targets = rows
            .iter()
            .map(|row| inputs.targets[*row])
            .collect::<Vec<_>>();
        let mut readout = LinearReadout::new(inputs.hv_dim, inputs.labels.len(), &mut rng);
        let trace = crate::readout_cuda::train_readout_sgd_cuda(
            &task_features,
            &task_targets,
            &mut readout,
            inputs.epochs,
            inputs.lr,
            inputs.samples_per_launch,
        )?;
        let weight = task_targets.len() as f32;
        weighted_loss += trace.loss * weight;
        weighted_accuracy += trace.accuracy * weight;
        total_rows += task_targets.len();
        heads.insert(task, readout);
    }

    let denom = total_rows.max(1) as f32;
    Ok(TaskHeadReadouts {
        heads,
        loss: weighted_loss / denom,
        accuracy: weighted_accuracy / denom,
    })
}

#[cfg(feature = "cuda")]
impl TaskResidualAdapter {
    fn new(input_dim: usize, output_dim: usize, rank: usize, rng: &mut impl rand::Rng) -> Self {
        let scale = (1.0 / input_dim.max(1) as f32).sqrt();
        let projection = (0..rank * input_dim)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        Self {
            input_dim,
            output_dim,
            rank,
            projection,
            weights: vec![0.0; output_dim * rank],
            bias: vec![0.0; output_dim],
        }
    }

    fn basis(&self, feature: &[f32]) -> Result<Vec<f32>> {
        if feature.len() != self.input_dim {
            return Err(SymError::Shape(format!(
                "residual adapter expected feature dim {}, got {}",
                self.input_dim,
                feature.len()
            )));
        }
        let mut basis = vec![0.0; self.rank];
        for (k, value) in basis.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (i, feature_value) in feature.iter().copied().enumerate() {
                acc += self.projection[k * self.input_dim + i] * feature_value;
            }
            *value = acc.tanh();
        }
        Ok(basis)
    }

    fn logits_for_row(&self, feature: &[f32], shared_logits: &[f32]) -> Result<Vec<f32>> {
        if shared_logits.len() != self.output_dim {
            return Err(SymError::Shape(format!(
                "residual adapter expected shared logits dim {}, got {}",
                self.output_dim,
                shared_logits.len()
            )));
        }
        let basis = self.basis(feature)?;
        let mut logits = shared_logits.to_vec();
        for (out, logit) in logits.iter_mut().enumerate() {
            let mut residual = self.bias[out];
            for (k, basis_value) in basis.iter().copied().enumerate() {
                residual += self.weights[out * self.rank + k] * basis_value;
            }
            *logit += residual;
        }
        Ok(logits)
    }
}

#[cfg(feature = "cuda")]
fn train_task_residual_adapters(
    inputs: ResidualAdapterTrainInputs<'_>,
) -> Result<TaskResidualAdapters> {
    use rand::SeedableRng;

    inputs
        .features
        .ensure_cols(inputs.hv_dim, "residual-adapter features")?;
    if inputs.cases.len() != inputs.features.rows || inputs.targets.len() != inputs.features.rows {
        return Err(SymError::Shape(format!(
            "residual-adapter rows mismatch: cases={} features={} targets={}",
            inputs.cases.len(),
            inputs.features.rows,
            inputs.targets.len()
        )));
    }
    if inputs.rank == 0 {
        return Err(SymError::InvalidArgument(
            "residual adapter rank must be greater than zero".to_string(),
        ));
    }

    let mut rows_by_task: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (row, case) in inputs.cases.iter().enumerate() {
        rows_by_task.entry(case.task.clone()).or_default().push(row);
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(inputs.seed);
    let mut adapters = BTreeMap::new();
    let mut weighted_loss = 0.0;
    let mut weighted_accuracy = 0.0;
    let mut total_rows = 0usize;

    for (task, rows) in rows_by_task {
        let task_features = select_feature_rows(inputs.features, &rows, inputs.hv_dim)?;
        let task_targets = rows
            .iter()
            .map(|row| inputs.targets[*row])
            .collect::<Vec<_>>();
        let mut adapter =
            TaskResidualAdapter::new(inputs.hv_dim, inputs.labels.len(), inputs.rank, &mut rng);
        let trace = train_residual_adapter_cpu(
            &task_features,
            &task_targets,
            inputs.shared_readout,
            &mut adapter,
            inputs.epochs,
            inputs.lr,
        )?;
        let weight = task_targets.len() as f32;
        weighted_loss += trace.loss * weight;
        weighted_accuracy += trace.accuracy * weight;
        total_rows += task_targets.len();
        adapters.insert(task, adapter);
    }

    let denom = total_rows.max(1) as f32;
    Ok(TaskResidualAdapters {
        adapters,
        rank: inputs.rank,
        loss: weighted_loss / denom,
        accuracy: weighted_accuracy / denom,
    })
}

#[cfg(feature = "cuda")]
fn train_residual_adapter_cpu(
    features: &Tensor,
    targets: &[usize],
    shared_readout: &symliquid_core::train::LinearReadout,
    adapter: &mut TaskResidualAdapter,
    epochs: usize,
    lr: f32,
) -> Result<symliquid_core::train::TrainingTrace> {
    features.ensure_cols(adapter.input_dim, "residual-adapter train features")?;
    if targets.len() != features.rows {
        return Err(SymError::Shape(format!(
            "residual-adapter train target batch expected {}, got {}",
            features.rows,
            targets.len()
        )));
    }
    if shared_readout.input_dim != adapter.input_dim
        || shared_readout.output_dim != adapter.output_dim
    {
        return Err(SymError::Shape(format!(
            "shared readout shape [{}, {}] incompatible with residual adapter [{}, {}]",
            shared_readout.input_dim,
            shared_readout.output_dim,
            adapter.input_dim,
            adapter.output_dim
        )));
    }
    if let Some(target) = targets
        .iter()
        .copied()
        .find(|target| *target >= adapter.output_dim)
    {
        return Err(SymError::InvalidArgument(format!(
            "target {target} outside residual adapter output dim {}",
            adapter.output_dim
        )));
    }

    let shared_logits = shared_readout.logits(features)?;
    let weight_decay = 1.0e-4;
    for _ in 0..epochs {
        for (row, target) in targets.iter().copied().enumerate() {
            let feature = features.row(row);
            let basis = adapter.basis(feature)?;
            let mut logits = shared_logits.row(row).to_vec();
            for (out, logit) in logits.iter_mut().enumerate() {
                let mut residual = adapter.bias[out];
                for (k, basis_value) in basis.iter().copied().enumerate() {
                    residual += adapter.weights[out * adapter.rank + k] * basis_value;
                }
                *logit += residual;
            }
            let probs = symliquid_core::tensor::softmax(&logits);
            for (out, prob) in probs.iter().copied().enumerate() {
                let delta = prob - if out == target { 1.0 } else { 0.0 };
                adapter.bias[out] -= lr * delta;
                for (k, basis_value) in basis.iter().copied().enumerate() {
                    let weight_index = out * adapter.rank + k;
                    let grad = delta * basis_value + weight_decay * adapter.weights[weight_index];
                    adapter.weights[weight_index] -= lr * grad;
                }
            }
        }
    }

    evaluate_residual_adapter_batch(features, targets, shared_readout, adapter)
}

#[cfg(feature = "cuda")]
fn evaluate_residual_adapter_batch(
    features: &Tensor,
    targets: &[usize],
    shared_readout: &symliquid_core::train::LinearReadout,
    adapter: &TaskResidualAdapter,
) -> Result<symliquid_core::train::TrainingTrace> {
    features.ensure_cols(adapter.input_dim, "residual-adapter eval features")?;
    if targets.len() != features.rows {
        return Err(SymError::Shape(format!(
            "residual-adapter eval target batch expected {}, got {}",
            features.rows,
            targets.len()
        )));
    }
    let shared_logits = shared_readout.logits(features)?;
    let mut loss = 0.0;
    let mut correct = 0usize;
    for (row, target) in targets.iter().copied().enumerate() {
        let logits = adapter.logits_for_row(features.row(row), shared_logits.row(row))?;
        let probs = symliquid_core::tensor::softmax(&logits);
        loss += -probs[target].max(1.0e-8).ln();
        correct += usize::from(symliquid_core::tensor::argmax(&probs) == target);
    }
    let scale = 1.0 / targets.len().max(1) as f32;
    Ok(symliquid_core::train::TrainingTrace {
        loss: loss * scale,
        accuracy: correct as f32 * scale,
        grad_norm: 0.0,
    })
}

#[cfg(feature = "cuda")]
fn evaluate_shared_readout(
    suite: &symliquid_core::benchmarks::BenchmarkSuite,
    features: &Tensor,
    labels: &[String],
    readout: &symliquid_core::train::LinearReadout,
    model_id: &str,
) -> Result<symliquid_core::benchmarks::BenchmarkReport> {
    use symliquid_core::benchmarks::{score_output, summarize, ModelResponse, RunMode};

    features.ensure_cols(readout.input_dim, "shared-head eval features")?;
    if features.rows != suite.cases.len() {
        return Err(SymError::Shape(format!(
            "shared-head eval rows mismatch: suite={} features={}",
            suite.cases.len(),
            features.rows
        )));
    }
    let logits = readout.logits(features)?;
    let mut results = Vec::with_capacity(suite.cases.len());
    for (row, case) in suite.cases.iter().enumerate() {
        let mut governed_logits = logits.row(row).to_vec();
        apply_governance_priors(case, labels, &mut governed_logits);
        let pred = governed_prediction(case, &governed_logits, labels);
        results.push(score_output(
            suite,
            case,
            ModelResponse {
                case_id: case.id.clone(),
                model_id: model_id.to_string(),
                mode: RunMode::SymLiquid,
                output: pred,
                runtime_ms: Some(0),
                token_count: Some(case.observation.split_whitespace().count()),
                tool_calls: Some(0),
                estimated_cost_usd: Some(0.0),
                notes: vec![
                    "Standalone SymLiquid CUDA rollout features with shared readout head and CGS governance priors."
                        .to_string(),
                ],
            },
        ));
    }

    Ok(symliquid_core::benchmarks::BenchmarkReport {
        summary: summarize(suite, model_id, RunMode::SymLiquid, &results),
        results,
    })
}

#[cfg(feature = "cuda")]
fn evaluate_task_head_readouts(
    suite: &symliquid_core::benchmarks::BenchmarkSuite,
    features: &Tensor,
    labels: &[String],
    heads: &BTreeMap<String, symliquid_core::train::LinearReadout>,
    model_id: &str,
) -> Result<symliquid_core::benchmarks::BenchmarkReport> {
    use symliquid_core::benchmarks::{score_output, summarize, ModelResponse, RunMode};

    features.ensure_cols(features.cols, "task-head eval features")?;
    if features.rows != suite.cases.len() {
        return Err(SymError::Shape(format!(
            "task-head eval rows mismatch: suite={} features={}",
            suite.cases.len(),
            features.rows
        )));
    }

    let mut results = Vec::with_capacity(suite.cases.len());
    for (row, case) in suite.cases.iter().enumerate() {
        let Some(head) = heads.get(&case.task) else {
            return Err(SymError::InvalidArgument(format!(
                "missing task readout head for '{}'",
                case.task
            )));
        };
        let feature = Tensor::new(1, features.cols, features.row(row).to_vec())?;
        let logits = head.logits(&feature)?;
        let mut governed_logits = logits.row(0).to_vec();
        apply_governance_priors(case, labels, &mut governed_logits);
        let pred = governed_prediction(case, &governed_logits, labels);
        results.push(score_output(
            suite,
            case,
            ModelResponse {
                case_id: case.id.clone(),
                model_id: model_id.to_string(),
                mode: RunMode::SymLiquid,
                output: pred,
                runtime_ms: Some(0),
                token_count: Some(case.observation.split_whitespace().count()),
                tool_calls: Some(0),
                estimated_cost_usd: Some(0.0),
                notes: vec![
                    "Standalone SymLiquid CUDA rollout features with task-specialized readout head."
                        .to_string(),
                ],
            },
        ));
    }

    Ok(symliquid_core::benchmarks::BenchmarkReport {
        summary: summarize(suite, model_id, RunMode::SymLiquid, &results),
        results,
    })
}

#[cfg(feature = "cuda")]
fn evaluate_residual_adapter_readouts(
    suite: &symliquid_core::benchmarks::BenchmarkSuite,
    features: &Tensor,
    labels: &[String],
    shared_readout: &symliquid_core::train::LinearReadout,
    adapters: &BTreeMap<String, TaskResidualAdapter>,
    model_id: &str,
) -> Result<symliquid_core::benchmarks::BenchmarkReport> {
    use symliquid_core::benchmarks::{score_output, summarize, ModelResponse, RunMode};

    features.ensure_cols(shared_readout.input_dim, "residual-adapter eval features")?;
    if features.rows != suite.cases.len() {
        return Err(SymError::Shape(format!(
            "residual-adapter eval rows mismatch: suite={} features={}",
            suite.cases.len(),
            features.rows
        )));
    }

    let shared_logits = shared_readout.logits(features)?;
    let mut results = Vec::with_capacity(suite.cases.len());
    for (row, case) in suite.cases.iter().enumerate() {
        let Some(adapter) = adapters.get(&case.task) else {
            return Err(SymError::InvalidArgument(format!(
                "missing residual adapter for '{}'",
                case.task
            )));
        };
        let mut logits = adapter.logits_for_row(features.row(row), shared_logits.row(row))?;
        apply_governance_priors(case, labels, &mut logits);
        let pred = governed_prediction(case, &logits, labels);
        results.push(score_output(
            suite,
            case,
            ModelResponse {
                case_id: case.id.clone(),
                model_id: model_id.to_string(),
                mode: RunMode::SymLiquid,
                output: pred,
                runtime_ms: Some(0),
                token_count: Some(case.observation.split_whitespace().count()),
                tool_calls: Some(0),
                estimated_cost_usd: Some(0.0),
                notes: vec![
                    "Standalone SymLiquid CUDA rollout features with shared readout, low-rank task residual adapters, and CGS governance priors."
                        .to_string(),
                ],
            },
        ));
    }

    Ok(symliquid_core::benchmarks::BenchmarkReport {
        summary: summarize(suite, model_id, RunMode::SymLiquid, &results),
        results,
    })
}

#[cfg(feature = "cuda")]
fn probe_linear_readout(
    cases: &[symliquid_core::benchmarks::BenchmarkCase],
    features: &Tensor,
    labels: &[String],
    readout: &symliquid_core::train::LinearReadout,
) -> Result<ReadoutProbeTrace> {
    features.ensure_cols(readout.input_dim, "shared-head probe features")?;
    if cases.len() != features.rows {
        return Err(SymError::Shape(format!(
            "shared-head probe rows mismatch: cases={} features={}",
            cases.len(),
            features.rows
        )));
    }
    let logits = readout.logits(features)?;
    let mut outcomes = Vec::with_capacity(cases.len());
    let mut correct = 0usize;
    for (row, case) in cases.iter().enumerate() {
        let mut governed_logits = logits.row(row).to_vec();
        apply_governance_priors(case, labels, &mut governed_logits);
        let pred = governed_prediction(case, &governed_logits, labels);
        let is_correct = pred.eq_ignore_ascii_case(&case.expected);
        if is_correct {
            correct += 1;
        }
        outcomes.push((case.task.clone(), is_correct));
    }
    Ok(ReadoutProbeTrace {
        accuracy: correct as f32 / cases.len().max(1) as f32,
        cases: cases.len(),
        task_breakdown: probe_task_breakdown(&outcomes),
    })
}

#[cfg(feature = "cuda")]
fn probe_task_head_readouts(
    cases: &[symliquid_core::benchmarks::BenchmarkCase],
    features: &Tensor,
    labels: &[String],
    heads: &BTreeMap<String, symliquid_core::train::LinearReadout>,
) -> Result<ReadoutProbeTrace> {
    if cases.len() != features.rows {
        return Err(SymError::Shape(format!(
            "task-head probe rows mismatch: cases={} features={}",
            cases.len(),
            features.rows
        )));
    }
    let mut outcomes = Vec::with_capacity(cases.len());
    let mut correct = 0usize;
    for (row, case) in cases.iter().enumerate() {
        let Some(head) = heads.get(&case.task) else {
            return Err(SymError::InvalidArgument(format!(
                "missing task readout head for '{}'",
                case.task
            )));
        };
        let feature = Tensor::new(1, features.cols, features.row(row).to_vec())?;
        let logits = head.logits(&feature)?;
        let mut governed_logits = logits.row(0).to_vec();
        apply_governance_priors(case, labels, &mut governed_logits);
        let pred = governed_prediction(case, &governed_logits, labels);
        let is_correct = pred.eq_ignore_ascii_case(&case.expected);
        if is_correct {
            correct += 1;
        }
        outcomes.push((case.task.clone(), is_correct));
    }
    Ok(ReadoutProbeTrace {
        accuracy: correct as f32 / cases.len().max(1) as f32,
        cases: cases.len(),
        task_breakdown: probe_task_breakdown(&outcomes),
    })
}

#[cfg(feature = "cuda")]
fn probe_residual_adapter_readouts(
    cases: &[symliquid_core::benchmarks::BenchmarkCase],
    features: &Tensor,
    labels: &[String],
    shared_readout: &symliquid_core::train::LinearReadout,
    adapters: &BTreeMap<String, TaskResidualAdapter>,
) -> Result<ReadoutProbeTrace> {
    features.ensure_cols(shared_readout.input_dim, "residual-adapter probe features")?;
    if cases.len() != features.rows {
        return Err(SymError::Shape(format!(
            "residual-adapter probe rows mismatch: cases={} features={}",
            cases.len(),
            features.rows
        )));
    }
    let shared_logits = shared_readout.logits(features)?;
    let mut outcomes = Vec::with_capacity(cases.len());
    let mut correct = 0usize;
    for (row, case) in cases.iter().enumerate() {
        let Some(adapter) = adapters.get(&case.task) else {
            return Err(SymError::InvalidArgument(format!(
                "missing residual adapter for '{}'",
                case.task
            )));
        };
        let mut logits = adapter.logits_for_row(features.row(row), shared_logits.row(row))?;
        apply_governance_priors(case, labels, &mut logits);
        let pred = governed_prediction(case, &logits, labels);
        let is_correct = pred.eq_ignore_ascii_case(&case.expected);
        if is_correct {
            correct += 1;
        }
        outcomes.push((case.task.clone(), is_correct));
    }
    Ok(ReadoutProbeTrace {
        accuracy: correct as f32 / cases.len().max(1) as f32,
        cases: cases.len(),
        task_breakdown: probe_task_breakdown(&outcomes),
    })
}

#[cfg(feature = "cuda")]
fn labeled_cases_and_targets(
    cases: &[symliquid_core::benchmarks::BenchmarkCase],
    labels: &[String],
) -> (Vec<symliquid_core::benchmarks::BenchmarkCase>, Vec<usize>) {
    let mut selected = Vec::new();
    let mut targets = Vec::new();
    for case in cases {
        if let Some(target) = labels.iter().position(|label| label == &case.expected) {
            selected.push(case.clone());
            targets.push(target);
        }
    }
    (selected, targets)
}

#[cfg(feature = "cuda")]
fn probe_rollout_params(
    fit_cases: &[symliquid_core::benchmarks::BenchmarkCase],
    validation_cases: &[symliquid_core::benchmarks::BenchmarkCase],
    labels: &[String],
    params: &RolloutParams,
    config: &RolloutFeatureConfig,
    probe: RolloutProbeConfig,
) -> Result<RolloutProbeTrace> {
    use rand::SeedableRng;
    use symliquid_core::train::LinearReadout;

    let (features, targets) = rollout_labeled_features_cuda(fit_cases, labels, params, config)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(probe.seed);
    let mut readout = LinearReadout::new(config.hv_dim, labels.len(), &mut rng);
    crate::readout_cuda::train_readout_sgd_cuda(
        &features,
        &targets,
        &mut readout,
        probe.epochs,
        probe.lr,
        probe.samples_per_launch,
    )?;
    let (validation_labeled, validation_targets) =
        labeled_cases_and_targets(validation_cases, labels);
    let validation_features = rollout_features_cuda(&validation_labeled, params, config)?;
    let raw_trace = readout.evaluate_batch(&validation_features, &validation_targets)?;
    let logits = readout.logits(&validation_features)?;
    let mut outcomes = Vec::with_capacity(validation_labeled.len());
    let mut correct = 0usize;
    for (row, case) in validation_labeled.iter().enumerate() {
        let mut governed_logits = logits.row(row).to_vec();
        apply_governance_priors(case, labels, &mut governed_logits);
        let pred = governed_prediction(case, &governed_logits, labels);
        let is_correct = pred.eq_ignore_ascii_case(&case.expected);
        if is_correct {
            correct += 1;
        }
        outcomes.push((case.task.clone(), is_correct));
    }
    let state_alignment = rollout_state_alignment(&validation_labeled, labels, params, config)?;
    Ok(RolloutProbeTrace {
        loss: raw_trace.loss,
        accuracy: correct as f32 / validation_labeled.len().max(1) as f32,
        state_alignment,
        cases: validation_labeled.len(),
        task_breakdown: probe_task_breakdown(&outcomes),
    })
}

#[cfg(feature = "cuda")]
fn probe_rollout_task_gated_params(inputs: TaskGatedProbeInputs<'_>) -> Result<RolloutProbeTrace> {
    use rand::SeedableRng;
    use symliquid_core::train::LinearReadout;

    let (features, targets) = rollout_labeled_features_task_gated_cuda(
        inputs.fit_cases,
        inputs.labels,
        inputs.base_params,
        inputs.candidate_params,
        inputs.candidate_tasks,
        inputs.config,
    )?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(inputs.probe.seed);
    let mut readout = LinearReadout::new(inputs.config.hv_dim, inputs.labels.len(), &mut rng);
    crate::readout_cuda::train_readout_sgd_cuda(
        &features,
        &targets,
        &mut readout,
        inputs.probe.epochs,
        inputs.probe.lr,
        inputs.probe.samples_per_launch,
    )?;
    let (validation_labeled, validation_targets) =
        labeled_cases_and_targets(inputs.validation_cases, inputs.labels);
    let validation_features = rollout_features_task_gated_cuda(
        &validation_labeled,
        inputs.base_params,
        inputs.candidate_params,
        inputs.candidate_tasks,
        inputs.config,
    )?;
    let raw_trace = readout.evaluate_batch(&validation_features, &validation_targets)?;
    let logits = readout.logits(&validation_features)?;
    let mut outcomes = Vec::with_capacity(validation_labeled.len());
    let mut correct = 0usize;
    for (row, case) in validation_labeled.iter().enumerate() {
        let mut governed_logits = logits.row(row).to_vec();
        apply_governance_priors(case, inputs.labels, &mut governed_logits);
        let pred = governed_prediction(case, &governed_logits, inputs.labels);
        let is_correct = pred.eq_ignore_ascii_case(&case.expected);
        if is_correct {
            correct += 1;
        }
        outcomes.push((case.task.clone(), is_correct));
    }
    let state_alignment = rollout_state_alignment_task_gated(
        &validation_labeled,
        inputs.labels,
        inputs.base_params,
        inputs.candidate_params,
        inputs.candidate_tasks,
        inputs.config,
    )?;
    Ok(RolloutProbeTrace {
        loss: raw_trace.loss,
        accuracy: correct as f32 / validation_labeled.len().max(1) as f32,
        state_alignment,
        cases: validation_labeled.len(),
        task_breakdown: probe_task_breakdown(&outcomes),
    })
}

#[cfg(feature = "cuda")]
fn probe_task_breakdown(
    outcomes: &[(String, bool)],
) -> BTreeMap<String, symliquid_core::benchmarks::TaskBreakdown> {
    let mut counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for (task, correct) in outcomes {
        let entry = counts.entry(task.clone()).or_insert((0, 0));
        entry.0 += 1;
        if *correct {
            entry.1 += 1;
        }
    }
    counts
        .into_iter()
        .map(|(task, (cases, correct))| {
            let accuracy = correct as f32 / cases.max(1) as f32;
            (
                task,
                symliquid_core::benchmarks::TaskBreakdown {
                    cases,
                    accuracy,
                    residual: 1.0 - accuracy,
                },
            )
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn candidate_state_task_gate(
    base: &BTreeMap<String, symliquid_core::benchmarks::TaskBreakdown>,
    candidate: &BTreeMap<String, symliquid_core::benchmarks::TaskBreakdown>,
) -> BTreeSet<String> {
    base.iter()
        .filter_map(|(task, base_breakdown)| {
            let candidate_breakdown = candidate.get(task)?;
            let delta = candidate_breakdown.accuracy - base_breakdown.accuracy;
            if delta >= 0.05 {
                Some(task.clone())
            } else {
                None
            }
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn worst_probe_task_accuracy_delta(
    base: &BTreeMap<String, symliquid_core::benchmarks::TaskBreakdown>,
    candidate: &BTreeMap<String, symliquid_core::benchmarks::TaskBreakdown>,
) -> f32 {
    if base.is_empty() {
        return 0.0;
    }
    base.iter()
        .map(|(task, base_breakdown)| {
            let candidate_accuracy = candidate
                .get(task)
                .map(|breakdown| breakdown.accuracy)
                .unwrap_or(0.0);
            candidate_accuracy - base_breakdown.accuracy
        })
        .fold(f32::INFINITY, f32::min)
}

#[cfg(feature = "cuda")]
fn rollout_state_alignment(
    cases: &[symliquid_core::benchmarks::BenchmarkCase],
    labels: &[String],
    params: &RolloutParams,
    config: &RolloutFeatureConfig,
) -> Result<f32> {
    if cases.is_empty() {
        return Ok(0.0);
    }
    let target_vectors = label_hypervectors(labels, config.hv_dim, "hv");
    let features = rollout_features_cuda(cases, params, config)?;
    let mut total = 0.0;
    let mut count = 0usize;
    for (row, case) in cases.iter().enumerate() {
        let Some(target) = labels.iter().position(|label| label == &case.expected) else {
            continue;
        };
        let feature_row = features.row(row);
        let target_row = &target_vectors[target];
        let dot = feature_row
            .iter()
            .zip(target_row)
            .map(|(left, right)| left * right)
            .sum::<f32>();
        let feature_norm = feature_row
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1.0e-6);
        let target_norm = (target_row.len() as f32).sqrt().max(1.0e-6);
        total += dot / (feature_norm * target_norm);
        count += 1;
    }
    Ok(total / count.max(1) as f32)
}

#[cfg(feature = "cuda")]
fn rollout_state_alignment_task_gated(
    cases: &[symliquid_core::benchmarks::BenchmarkCase],
    labels: &[String],
    base_params: &RolloutParams,
    candidate_params: &RolloutParams,
    candidate_tasks: &BTreeSet<String>,
    config: &RolloutFeatureConfig,
) -> Result<f32> {
    if cases.is_empty() {
        return Ok(0.0);
    }
    let target_vectors = label_hypervectors(labels, config.hv_dim, "hv");
    let features = rollout_features_task_gated_cuda(
        cases,
        base_params,
        candidate_params,
        candidate_tasks,
        config,
    )?;
    let mut total = 0.0;
    let mut count = 0usize;
    for (row, case) in cases.iter().enumerate() {
        let Some(target) = labels.iter().position(|label| label == &case.expected) else {
            continue;
        };
        let feature_row = features.row(row);
        let target_row = &target_vectors[target];
        let dot = feature_row
            .iter()
            .zip(target_row)
            .map(|(left, right)| left * right)
            .sum::<f32>();
        let feature_norm = feature_row
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1.0e-6);
        let target_norm = (target_row.len() as f32).sqrt().max(1.0e-6);
        total += dot / (feature_norm * target_norm);
        count += 1;
    }
    Ok(total / count.max(1) as f32)
}

#[cfg(feature = "cuda")]
fn train_state_params_from_rollouts(
    cases: &[symliquid_core::benchmarks::BenchmarkCase],
    labels: &[String],
    params: &mut RolloutParams,
    config: &RolloutFeatureConfig,
    epochs: usize,
    lr: f32,
) -> Result<()> {
    let hidden_targets = label_hypervectors(labels, config.hidden_dim, "hidden");
    let reservoir_targets = label_hypervectors(labels, config.reservoir_dim, "reservoir");
    let hv_targets = label_hypervectors(labels, config.hv_dim, "hv");
    let labeled = cases
        .iter()
        .filter_map(|case| {
            labels
                .iter()
                .position(|label| label == &case.expected)
                .map(|target| (case.clone(), target))
        })
        .collect::<Vec<_>>();

    for _ in 0..epochs {
        for chunk in labeled.chunks(config.rollout_batch.max(1)) {
            let chunk_cases = chunk
                .iter()
                .map(|(case, _)| case.clone())
                .collect::<Vec<_>>();
            let targets = chunk.iter().map(|(_, target)| *target).collect::<Vec<_>>();
            let state = rollout_state_for_cases_cuda(&chunk_cases, params, config)?;
            for (row, target) in targets.iter().copied().enumerate() {
                let obs_features = pooled_observation_features(&chunk_cases[row], config);
                update_state_dynamics(
                    StateDynamicsSample {
                        obs_features: &obs_features,
                        hidden_state: state.h.row(row),
                        reservoir_state: state.r.row(row),
                        hidden_target: &hidden_targets[target],
                        reservoir_target: &reservoir_targets[target],
                    },
                    params,
                    config,
                    lr * 0.15,
                );
                update_hv_projection(
                    state.r.row(row),
                    &hv_targets[target],
                    params,
                    config.reservoir_dim,
                    lr,
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn rollout_state_for_cases_cuda(
    cases: &[symliquid_core::benchmarks::BenchmarkCase],
    params: &RolloutParams,
    config: &RolloutFeatureConfig,
) -> Result<RolloutState> {
    let rollout = RolloutConfig {
        batch: cases.len(),
        obs_dim: config.obs_dim,
        hidden_dim: config.hidden_dim,
        reservoir_dim: config.reservoir_dim,
        hv_dim: config.hv_dim,
        dt: config.dt,
        alpha: config.alpha,
        memory_decay: config.memory_decay,
    };
    let observations = encode_observation_batch(cases, config)?;
    let state = RolloutState {
        h: Tensor::zeros(cases.len(), config.hidden_dim),
        r: Tensor::zeros(cases.len(), config.reservoir_dim),
        memory: Tensor::zeros(cases.len(), config.hv_dim),
    };
    rollout_sequence_cuda(&observations, &state, params, &rollout)
}

#[cfg(feature = "cuda")]
fn rollout_features_cuda(
    cases: &[symliquid_core::benchmarks::BenchmarkCase],
    params: &RolloutParams,
    config: &RolloutFeatureConfig,
) -> Result<Tensor> {
    let labels = symliquid_core::benchmarks::standalone_output_vocab();
    let label_vectors = label_hypervectors(&labels, config.hv_dim, "hv");
    let mut features = Vec::with_capacity(cases.len() * config.hv_dim);
    for chunk in cases.chunks(config.rollout_batch.max(1)) {
        let final_state = rollout_state_for_cases_cuda(chunk, params, config)?;
        for (row, case) in chunk.iter().enumerate() {
            let mut feature = final_state.memory.row(row).to_vec();
            if let Some(memory_feature) =
                entity_slot_vsa_feature(case, &labels, &label_vectors, config.hv_dim)
            {
                let scale = config.seq_len as f32 * 0.45;
                for (value, memory_value) in feature.iter_mut().zip(memory_feature) {
                    *value = *value * 0.55 + scale * memory_value;
                }
            }
            features.extend_from_slice(&feature);
        }
    }
    Tensor::new(cases.len(), config.hv_dim, features)
}

#[cfg(feature = "cuda")]
fn rollout_features_task_gated_cuda(
    cases: &[symliquid_core::benchmarks::BenchmarkCase],
    base_params: &RolloutParams,
    candidate_params: &RolloutParams,
    candidate_tasks: &BTreeSet<String>,
    config: &RolloutFeatureConfig,
) -> Result<Tensor> {
    let mut base_cases = Vec::new();
    let mut base_indices = Vec::new();
    let mut candidate_cases = Vec::new();
    let mut candidate_indices = Vec::new();
    for (idx, case) in cases.iter().enumerate() {
        if candidate_tasks.contains(&case.task) {
            candidate_cases.push(case.clone());
            candidate_indices.push(idx);
        } else {
            base_cases.push(case.clone());
            base_indices.push(idx);
        }
    }
    let mut features = vec![0.0; cases.len() * config.hv_dim];
    if !base_cases.is_empty() {
        let base_features = rollout_features_cuda(&base_cases, base_params, config)?;
        copy_task_gated_feature_rows(&base_features, &base_indices, &mut features, config.hv_dim);
    }
    if !candidate_cases.is_empty() {
        let candidate_features = rollout_features_cuda(&candidate_cases, candidate_params, config)?;
        copy_task_gated_feature_rows(
            &candidate_features,
            &candidate_indices,
            &mut features,
            config.hv_dim,
        );
    }
    Tensor::new(cases.len(), config.hv_dim, features)
}

#[cfg(feature = "cuda")]
fn copy_task_gated_feature_rows(
    source: &Tensor,
    target_indices: &[usize],
    target: &mut [f32],
    hv_dim: usize,
) {
    for (source_row, target_row) in target_indices.iter().copied().enumerate() {
        let target_start = target_row * hv_dim;
        target[target_start..target_start + hv_dim].copy_from_slice(source.row(source_row));
    }
}

#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
fn init_rollout_params(config: &RolloutFeatureConfig, rng: &mut impl rand::Rng) -> RolloutParams {
    RolloutParams {
        obs_to_h: uniform_vec(config.hidden_dim * config.obs_dim, 0.18, rng),
        h_recurrent: uniform_vec(config.hidden_dim * config.hidden_dim, 0.04, rng),
        h_bias: uniform_vec(config.hidden_dim, 0.03, rng),
        reservoir_input: uniform_vec(config.reservoir_dim * config.hidden_dim, 0.14, rng),
        reservoir_recurrent: uniform_vec(config.reservoir_dim * config.reservoir_dim, 0.02, rng),
        reservoir_bias: uniform_vec(config.reservoir_dim, 0.02, rng),
        hv_proj: uniform_vec(config.hv_dim * config.reservoir_dim, 0.1, rng),
    }
}

#[cfg(feature = "cuda")]
fn update_state_dynamics(
    sample: StateDynamicsSample<'_>,
    params: &mut RolloutParams,
    config: &RolloutFeatureConfig,
    lr: f32,
) {
    update_linear_state_block(StateBlockUpdate {
        weights: &mut params.obs_to_h,
        bias: Some(&mut params.h_bias),
        rows: config.hidden_dim,
        cols: config.obs_dim,
        input: sample.obs_features,
        output: sample.hidden_state,
        target: sample.hidden_target,
        lr,
        max_abs: 0.8,
    });
    update_linear_state_block(StateBlockUpdate {
        weights: &mut params.h_recurrent,
        bias: None,
        rows: config.hidden_dim,
        cols: config.hidden_dim,
        input: sample.hidden_state,
        output: sample.hidden_state,
        target: sample.hidden_target,
        lr: lr * 0.25,
        max_abs: 0.25,
    });
    update_linear_state_block(StateBlockUpdate {
        weights: &mut params.reservoir_input,
        bias: Some(&mut params.reservoir_bias),
        rows: config.reservoir_dim,
        cols: config.hidden_dim,
        input: sample.hidden_state,
        output: sample.reservoir_state,
        target: sample.reservoir_target,
        lr,
        max_abs: 0.8,
    });
    update_linear_state_block(StateBlockUpdate {
        weights: &mut params.reservoir_recurrent,
        bias: None,
        rows: config.reservoir_dim,
        cols: config.reservoir_dim,
        input: sample.reservoir_state,
        output: sample.reservoir_state,
        target: sample.reservoir_target,
        lr: lr * 0.2,
        max_abs: 0.2,
    });
}

#[cfg(feature = "cuda")]
fn update_linear_state_block(mut update: StateBlockUpdate<'_>) {
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

#[cfg(feature = "cuda")]
fn update_hv_projection(
    reservoir_state: &[f32],
    target_hv: &[f32],
    params: &mut RolloutParams,
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

#[cfg(feature = "cuda")]
#[cfg(not(feature = "cuda"))]
pub fn rollout_sequence_cuda(
    _observations: &Tensor,
    _state: &RolloutState,
    _params: &RolloutParams,
    _config: &RolloutConfig,
) -> Result<RolloutState> {
    Err(SymError::InvalidArgument(
        "CUDA feature is not enabled for rollout state updates".to_string(),
    ))
}

fn validate_rollout(
    observations: &Tensor,
    state: &RolloutState,
    params: &RolloutParams,
    config: &RolloutConfig,
) -> Result<()> {
    if config.batch == 0
        || config.obs_dim == 0
        || config.hidden_dim == 0
        || config.reservoir_dim == 0
        || config.hv_dim == 0
    {
        return Err(SymError::InvalidArgument(
            "rollout config dimensions must be nonzero".to_string(),
        ));
    }
    observations.ensure_cols(config.obs_dim, "rollout observations")?;
    if !observations.rows.is_multiple_of(config.batch) {
        return Err(SymError::Shape(format!(
            "rollout observations rows {} must be divisible by batch {}",
            observations.rows, config.batch
        )));
    }
    state.h.ensure_cols(config.hidden_dim, "rollout h")?;
    state.r.ensure_cols(config.reservoir_dim, "rollout r")?;
    state.memory.ensure_cols(config.hv_dim, "rollout memory")?;
    if state.h.rows != config.batch
        || state.r.rows != config.batch
        || state.memory.rows != config.batch
    {
        return Err(SymError::Shape(format!(
            "rollout state rows must equal batch {}",
            config.batch
        )));
    }
    expect_len(
        "obs_to_h",
        params.obs_to_h.len(),
        config.hidden_dim * config.obs_dim,
    )?;
    expect_len(
        "h_recurrent",
        params.h_recurrent.len(),
        config.hidden_dim * config.hidden_dim,
    )?;
    expect_len("h_bias", params.h_bias.len(), config.hidden_dim)?;
    expect_len(
        "reservoir_input",
        params.reservoir_input.len(),
        config.reservoir_dim * config.hidden_dim,
    )?;
    expect_len(
        "reservoir_recurrent",
        params.reservoir_recurrent.len(),
        config.reservoir_dim * config.reservoir_dim,
    )?;
    expect_len(
        "reservoir_bias",
        params.reservoir_bias.len(),
        config.reservoir_dim,
    )?;
    expect_len(
        "hv_proj",
        params.hv_proj.len(),
        config.hv_dim * config.reservoir_dim,
    )?;
    Ok(())
}

fn expect_len(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(SymError::Shape(format!(
            "{name} expected length {expected}, got {actual}"
        )))
    }
}

#[cfg(feature = "cuda")]
fn cuda_error(error: impl std::fmt::Display) -> SymError {
    SymError::InvalidArgument(format!("CUDA rollout operation failed: {error}"))
}
