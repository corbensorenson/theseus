use std::ffi::c_void;

use metal::{CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use symliquid_core::error::{Result, SymError};
use symliquid_core::tensor::Tensor;
use symliquid_core::train::{LinearReadout, TrainingTrace};
use symliquid_cuda::rollout_cuda::{RolloutConfig, RolloutParams, RolloutState};

const ROLLOUT_METALLIB: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/rollout_state_update.metallib"));

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalRolloutConfig {
    observation_offset: i32,
    obs_dim: i32,
    hidden_dim: i32,
    reservoir_dim: i32,
    hv_dim: i32,
    dt: f32,
    alpha: f32,
    memory_decay: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalLinearReadoutConfig {
    sample_count: i32,
    input_dim: i32,
    output_dim: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalReadoutSgdConfig {
    sample_start: i32,
    sample_count: i32,
    input_dim: i32,
    output_dim: i32,
    lr: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalReadoutBagSgdConfig {
    sample_start: i32,
    sample_count: i32,
    input_dim: i32,
    output_dim: i32,
    targets_per_sample: i32,
    lr: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalStateLinearUpdateConfig {
    rows: i32,
    cols: i32,
    lr: f32,
    max_abs: f32,
    has_bias: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalStateHvProjectionUpdateConfig {
    hv_dim: i32,
    reservoir_dim: i32,
    lr: f32,
}

pub fn metal_available() -> bool {
    Device::system_default().is_some()
}

pub fn train_readout_sgd_metal(
    features: &Tensor,
    targets: &[usize],
    readout: &mut LinearReadout,
    epochs: usize,
    lr: f32,
    samples_per_launch: usize,
) -> Result<TrainingTrace> {
    features.ensure_cols(readout.input_dim, "Metal readout training features")?;
    if targets.len() != features.rows {
        return Err(SymError::Shape(format!(
            "target batch expected {}, got {}",
            features.rows,
            targets.len()
        )));
    }
    if readout.output_dim == 0 {
        return Err(SymError::InvalidArgument(
            "Metal readout trainer output_dim must be nonzero".to_string(),
        ));
    }
    if readout.output_dim > 256 {
        return Err(SymError::InvalidArgument(format!(
            "Metal readout trainer supports output_dim <= 256, got {}",
            readout.output_dim
        )));
    }
    if let Some(target) = targets
        .iter()
        .copied()
        .find(|target| *target >= readout.output_dim)
    {
        return Err(SymError::InvalidArgument(format!(
            "target {target} outside output dim {}",
            readout.output_dim
        )));
    }
    if features.rows == 0 || epochs == 0 {
        return readout.evaluate_batch(features, targets);
    }

    let device = Device::system_default().ok_or_else(|| {
        SymError::InvalidArgument("no default Metal device available".to_string())
    })?;
    let queue = device.new_command_queue();
    let pipeline = kernel_pipeline(&device, "readout_sgd_samples_kernel")?;
    let features_buf = shared_buffer_with_data(&device, &features.data);
    let targets_i32 = targets
        .iter()
        .copied()
        .map(|target| target as i32)
        .collect::<Vec<_>>();
    let targets_buf = shared_i32_buffer_with_data(&device, &targets_i32);
    let weights_buf = shared_buffer_with_data(&device, &readout.weights);
    let bias_buf = shared_buffer_with_data(&device, &readout.bias);
    let chunk = samples_per_launch.clamp(1, features.rows.max(1));

    for _ in 0..epochs {
        for sample_start in (0..features.rows).step_by(chunk) {
            let sample_count = (features.rows - sample_start).min(chunk);
            encode_readout_sgd_chunk(
                &device,
                &queue,
                &pipeline,
                &features_buf,
                &targets_buf,
                &weights_buf,
                &bias_buf,
                MetalReadoutSgdConfig {
                    sample_start: sample_start as i32,
                    sample_count: sample_count as i32,
                    input_dim: readout.input_dim as i32,
                    output_dim: readout.output_dim as i32,
                    lr,
                },
            )?;
        }
    }

    readout.weights = read_f32_buffer(&weights_buf, readout.weights.len());
    readout.bias = read_f32_buffer(&bias_buf, readout.bias.len());
    readout.evaluate_batch(features, targets)
}

pub fn train_readout_bag_sgd_metal(
    features: &Tensor,
    target_bags: &[usize],
    targets_per_sample: usize,
    readout: &mut LinearReadout,
    epochs: usize,
    lr: f32,
    samples_per_launch: usize,
) -> Result<TrainingTrace> {
    features.ensure_cols(readout.input_dim, "Metal bag readout features")?;
    if targets_per_sample == 0 {
        return Err(SymError::InvalidArgument(
            "targets_per_sample must be greater than zero".to_string(),
        ));
    }
    if target_bags.len() != features.rows.saturating_mul(targets_per_sample) {
        return Err(SymError::Shape(format!(
            "target bag batch expected {}, got {}",
            features.rows.saturating_mul(targets_per_sample),
            target_bags.len()
        )));
    }
    if readout.output_dim == 0 {
        return Err(SymError::InvalidArgument(
            "Metal bag readout trainer output_dim must be nonzero".to_string(),
        ));
    }
    if readout.output_dim > 256 {
        return Err(SymError::InvalidArgument(format!(
            "Metal bag readout trainer supports output_dim <= 256, got {}",
            readout.output_dim
        )));
    }
    if let Some(target) = target_bags
        .iter()
        .copied()
        .find(|target| *target >= readout.output_dim)
    {
        return Err(SymError::InvalidArgument(format!(
            "target {target} outside output dim {}",
            readout.output_dim
        )));
    }
    if features.rows == 0 || epochs == 0 {
        return readout.train_batch_target_bags(features, target_bags, targets_per_sample, 0.0);
    }

    let device = Device::system_default().ok_or_else(|| {
        SymError::InvalidArgument("no default Metal device available".to_string())
    })?;
    let queue = device.new_command_queue();
    let pipeline = kernel_pipeline(&device, "readout_bag_sgd_samples_kernel")?;
    let features_buf = shared_buffer_with_data(&device, &features.data);
    let target_bags_i32 = target_bags
        .iter()
        .copied()
        .map(|target| target as i32)
        .collect::<Vec<_>>();
    let target_bags_buf = shared_i32_buffer_with_data(&device, &target_bags_i32);
    let weights_buf = shared_buffer_with_data(&device, &readout.weights);
    let bias_buf = shared_buffer_with_data(&device, &readout.bias);
    let chunk = samples_per_launch.clamp(1, features.rows.max(1));

    for _ in 0..epochs {
        for sample_start in (0..features.rows).step_by(chunk) {
            let sample_count = (features.rows - sample_start).min(chunk);
            encode_readout_bag_sgd_chunk(
                &device,
                &queue,
                &pipeline,
                &features_buf,
                &target_bags_buf,
                &weights_buf,
                &bias_buf,
                MetalReadoutBagSgdConfig {
                    sample_start: sample_start as i32,
                    sample_count: sample_count as i32,
                    input_dim: readout.input_dim as i32,
                    output_dim: readout.output_dim as i32,
                    targets_per_sample: targets_per_sample as i32,
                    lr,
                },
            )?;
        }
    }

    readout.weights = read_f32_buffer(&weights_buf, readout.weights.len());
    readout.bias = read_f32_buffer(&bias_buf, readout.bias.len());
    readout.train_batch_target_bags(features, target_bags, targets_per_sample, 0.0)
}

#[allow(clippy::too_many_arguments)]
pub fn update_state_dynamics_sample_metal(
    obs_features: &[f32],
    hidden_state: &[f32],
    reservoir_state: &[f32],
    hidden_target: &[f32],
    reservoir_target: &[f32],
    params: &mut RolloutParams,
    config: &RolloutConfig,
    lr: f32,
) -> Result<()> {
    expected_len(obs_features, config.obs_dim, "state obs_features")?;
    expected_len(hidden_state, config.hidden_dim, "state hidden_state")?;
    expected_len(
        reservoir_state,
        config.reservoir_dim,
        "state reservoir_state",
    )?;
    expected_len(hidden_target, config.hidden_dim, "state hidden_target")?;
    expected_len(
        reservoir_target,
        config.reservoir_dim,
        "state reservoir_target",
    )?;

    update_state_linear_block_metal(
        &mut params.obs_to_h,
        Some(&mut params.h_bias),
        config.hidden_dim,
        config.obs_dim,
        obs_features,
        hidden_state,
        hidden_target,
        lr,
        0.8,
    )?;
    update_state_linear_block_metal(
        &mut params.h_recurrent,
        None,
        config.hidden_dim,
        config.hidden_dim,
        hidden_state,
        hidden_state,
        hidden_target,
        lr * 0.25,
        0.25,
    )?;
    update_state_linear_block_metal(
        &mut params.reservoir_input,
        Some(&mut params.reservoir_bias),
        config.reservoir_dim,
        config.hidden_dim,
        hidden_state,
        reservoir_state,
        reservoir_target,
        lr,
        0.8,
    )?;
    update_state_linear_block_metal(
        &mut params.reservoir_recurrent,
        None,
        config.reservoir_dim,
        config.reservoir_dim,
        reservoir_state,
        reservoir_state,
        reservoir_target,
        lr * 0.2,
        0.2,
    )
}

pub fn update_hv_projection_metal(
    reservoir_state: &[f32],
    target_hv: &[f32],
    params: &mut RolloutParams,
    reservoir_dim: usize,
    lr: f32,
) -> Result<()> {
    expected_len(reservoir_state, reservoir_dim, "state reservoir_state")?;
    if reservoir_dim == 0 {
        return Err(SymError::InvalidArgument(
            "state reservoir_dim must be nonzero".to_string(),
        ));
    }
    if target_hv.is_empty() {
        return Err(SymError::InvalidArgument(
            "state target_hv must be nonempty".to_string(),
        ));
    }
    expected_len(
        &params.hv_proj,
        target_hv.len().saturating_mul(reservoir_dim),
        "hv_proj",
    )?;

    let device = Device::system_default().ok_or_else(|| {
        SymError::InvalidArgument("no default Metal device available".to_string())
    })?;
    let queue = device.new_command_queue();
    let pipeline = kernel_pipeline(&device, "state_hv_projection_update_kernel")?;
    let weights_buf = shared_buffer_with_data(&device, &params.hv_proj);
    let reservoir_buf = shared_buffer_with_data(&device, reservoir_state);
    let target_buf = shared_buffer_with_data(&device, target_hv);
    let cfg_buf = shared_bytes(
        &device,
        &MetalStateHvProjectionUpdateConfig {
            hv_dim: target_hv.len() as i32,
            reservoir_dim: reservoir_dim as i32,
            lr,
        },
    );
    encode_state_hv_projection_update(
        &queue,
        &pipeline,
        &weights_buf,
        &reservoir_buf,
        &target_buf,
        &cfg_buf,
        target_hv.len().saturating_mul(reservoir_dim),
    )?;
    params.hv_proj = read_f32_buffer(&weights_buf, params.hv_proj.len());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn update_state_linear_block_metal(
    weights: &mut Vec<f32>,
    bias: Option<&mut Vec<f32>>,
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &[f32],
    target: &[f32],
    lr: f32,
    max_abs: f32,
) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(SymError::InvalidArgument(
            "state linear update dimensions must be nonzero".to_string(),
        ));
    }
    expected_len(weights, rows.saturating_mul(cols), "state linear weights")?;
    expected_len(input, cols, "state linear input")?;
    expected_len(output, rows, "state linear output")?;
    expected_len(target, rows, "state linear target")?;

    let device = Device::system_default().ok_or_else(|| {
        SymError::InvalidArgument("no default Metal device available".to_string())
    })?;
    let queue = device.new_command_queue();
    let pipeline = kernel_pipeline(&device, "state_linear_update_kernel")?;
    let weights_buf = shared_buffer_with_data(&device, weights);
    let mut dummy_bias = vec![0.0f32; rows];
    let (bias_len, bias_buf) = if let Some(values) = bias.as_deref() {
        expected_len(values, rows, "state linear bias")?;
        (values.len(), shared_buffer_with_data(&device, values))
    } else {
        (
            dummy_bias.len(),
            shared_buffer_with_data(&device, &dummy_bias),
        )
    };
    let input_buf = shared_buffer_with_data(&device, input);
    let output_buf = shared_buffer_with_data(&device, output);
    let target_buf = shared_buffer_with_data(&device, target);
    let cfg_buf = shared_bytes(
        &device,
        &MetalStateLinearUpdateConfig {
            rows: rows as i32,
            cols: cols as i32,
            lr,
            max_abs,
            has_bias: if bias.is_some() { 1 } else { 0 },
        },
    );
    encode_state_linear_update(
        &queue,
        &pipeline,
        &weights_buf,
        &bias_buf,
        &input_buf,
        &output_buf,
        &target_buf,
        &cfg_buf,
        rows.saturating_mul(cols),
    )?;
    *weights = read_f32_buffer(&weights_buf, weights.len());
    if let Some(values) = bias {
        *values = read_f32_buffer(&bias_buf, bias_len);
    } else {
        dummy_bias = read_f32_buffer(&bias_buf, bias_len);
        debug_assert_eq!(dummy_bias.len(), rows);
    }
    Ok(())
}

pub fn linear_readout_logits_metal(features: &Tensor, readout: &LinearReadout) -> Result<Tensor> {
    features.ensure_cols(readout.input_dim, "Metal linear readout features")?;
    if readout.output_dim == 0 {
        return Err(SymError::InvalidArgument(
            "Metal linear readout output_dim must be nonzero".to_string(),
        ));
    }
    if readout.weights.len() != readout.input_dim.saturating_mul(readout.output_dim) {
        return Err(SymError::Shape(format!(
            "Metal linear readout expected {} weights, got {}",
            readout.input_dim.saturating_mul(readout.output_dim),
            readout.weights.len()
        )));
    }
    if readout.bias.len() != readout.output_dim {
        return Err(SymError::Shape(format!(
            "Metal linear readout expected {} bias values, got {}",
            readout.output_dim,
            readout.bias.len()
        )));
    }
    if features.rows == 0 {
        return Tensor::new(0, readout.output_dim, Vec::new());
    }

    let device = Device::system_default().ok_or_else(|| {
        SymError::InvalidArgument("no default Metal device available".to_string())
    })?;
    let queue = device.new_command_queue();
    let pipeline = kernel_pipeline(&device, "linear_readout_logits_kernel")?;
    let features_buf = shared_buffer_with_data(&device, &features.data);
    let weights_buf = shared_buffer_with_data(&device, &readout.weights);
    let bias_buf = shared_buffer_with_data(&device, &readout.bias);
    let logits_len = features.rows.saturating_mul(readout.output_dim);
    let logits_buf = shared_zero_buffer(&device, logits_len);
    let cfg = MetalLinearReadoutConfig {
        sample_count: features.rows as i32,
        input_dim: readout.input_dim as i32,
        output_dim: readout.output_dim as i32,
    };
    let cfg_buf = shared_bytes(&device, &cfg);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    for (idx, buffer) in [
        &features_buf,
        &weights_buf,
        &bias_buf,
        &logits_buf,
        &cfg_buf,
    ]
    .iter()
    .enumerate()
    {
        encoder.set_buffer(idx as u64, Some(buffer), 0);
    }
    let threads = pipeline.max_total_threads_per_threadgroup().min(256).max(1);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: logits_len as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Tensor::new(
        features.rows,
        readout.output_dim,
        read_f32_buffer(&logits_buf, logits_len),
    )
}

pub fn rollout_sequence_metal(
    observations: &Tensor,
    state: &RolloutState,
    params: &RolloutParams,
    config: &RolloutConfig,
) -> Result<RolloutState> {
    validate_rollout(observations, state, params, config)?;
    let steps = observations.rows / config.batch;
    let device = Device::system_default().ok_or_else(|| {
        SymError::InvalidArgument("no default Metal device available".to_string())
    })?;
    let queue = device.new_command_queue();
    let pipeline = rollout_pipeline(&device)?;

    let observations_buf = shared_buffer_with_data(&device, &observations.data);
    let obs_to_h_buf = shared_buffer_with_data(&device, &params.obs_to_h);
    let h_recurrent_buf = shared_buffer_with_data(&device, &params.h_recurrent);
    let h_bias_buf = shared_buffer_with_data(&device, &params.h_bias);
    let reservoir_input_buf = shared_buffer_with_data(&device, &params.reservoir_input);
    let reservoir_recurrent_buf = shared_buffer_with_data(&device, &params.reservoir_recurrent);
    let reservoir_bias_buf = shared_buffer_with_data(&device, &params.reservoir_bias);
    let hv_proj_buf = shared_buffer_with_data(&device, &params.hv_proj);

    let mut h_current = shared_buffer_with_data(&device, &state.h.data);
    let mut h_next = shared_zero_buffer(&device, state.h.data.len());
    let mut r_current = shared_buffer_with_data(&device, &state.r.data);
    let mut r_next = shared_zero_buffer(&device, state.r.data.len());
    let mut memory_current = shared_buffer_with_data(&device, &state.memory.data);
    let mut memory_next = shared_zero_buffer(&device, state.memory.data.len());

    for step in 0..steps {
        let cfg = MetalRolloutConfig {
            observation_offset: (step * config.batch * config.obs_dim) as i32,
            obs_dim: config.obs_dim as i32,
            hidden_dim: config.hidden_dim as i32,
            reservoir_dim: config.reservoir_dim as i32,
            hv_dim: config.hv_dim as i32,
            dt: config.dt,
            alpha: config.alpha,
            memory_decay: config.memory_decay,
        };
        let cfg_buf = shared_bytes(&device, &cfg);
        encode_rollout_step(
            &queue,
            &pipeline,
            config,
            &observations_buf,
            &h_current,
            &r_current,
            &memory_current,
            &h_next,
            &r_next,
            &memory_next,
            &obs_to_h_buf,
            &h_recurrent_buf,
            &h_bias_buf,
            &reservoir_input_buf,
            &reservoir_recurrent_buf,
            &reservoir_bias_buf,
            &hv_proj_buf,
            &cfg_buf,
        )?;
        std::mem::swap(&mut h_current, &mut h_next);
        std::mem::swap(&mut r_current, &mut r_next);
        std::mem::swap(&mut memory_current, &mut memory_next);
    }

    Ok(RolloutState {
        h: Tensor::new(
            config.batch,
            config.hidden_dim,
            read_f32_buffer(&h_current, state.h.data.len()),
        )?,
        r: Tensor::new(
            config.batch,
            config.reservoir_dim,
            read_f32_buffer(&r_current, state.r.data.len()),
        )?,
        memory: Tensor::new(
            config.batch,
            config.hv_dim,
            read_f32_buffer(&memory_current, state.memory.data.len()),
        )?,
    })
}

fn rollout_pipeline(device: &Device) -> Result<ComputePipelineState> {
    kernel_pipeline(device, "rollout_state_update_kernel")
}

fn kernel_pipeline(device: &Device, function_name: &str) -> Result<ComputePipelineState> {
    let library = device
        .new_library_with_data(ROLLOUT_METALLIB)
        .map_err(|error| {
            SymError::InvalidArgument(format!("failed to load rollout Metal library: {error}"))
        })?;
    let function = library.get_function(function_name, None).map_err(|error| {
        SymError::InvalidArgument(format!(
            "failed to load Metal function {function_name}: {error}"
        ))
    })?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| {
            SymError::InvalidArgument(format!(
                "failed to build Metal pipeline {function_name}: {error}"
            ))
        })
}

#[allow(clippy::too_many_arguments)]
fn encode_rollout_step(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    config: &RolloutConfig,
    observations: &metal::Buffer,
    h_current: &metal::Buffer,
    r_current: &metal::Buffer,
    memory_current: &metal::Buffer,
    h_next: &metal::Buffer,
    r_next: &metal::Buffer,
    memory_next: &metal::Buffer,
    obs_to_h: &metal::Buffer,
    h_recurrent: &metal::Buffer,
    h_bias: &metal::Buffer,
    reservoir_input: &metal::Buffer,
    reservoir_recurrent: &metal::Buffer,
    reservoir_bias: &metal::Buffer,
    hv_proj: &metal::Buffer,
    cfg: &metal::Buffer,
) -> Result<()> {
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (idx, buffer) in [
        observations,
        h_current,
        r_current,
        memory_current,
        h_next,
        r_next,
        memory_next,
        obs_to_h,
        h_recurrent,
        h_bias,
        reservoir_input,
        reservoir_recurrent,
        reservoir_bias,
        hv_proj,
        cfg,
    ]
    .iter()
    .enumerate()
    {
        encoder.set_buffer(idx as u64, Some(buffer), 0);
    }
    let threads = pipeline.max_total_threads_per_threadgroup().min(256).max(1);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: config.batch as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

fn encode_readout_sgd_chunk(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    features: &metal::Buffer,
    targets: &metal::Buffer,
    weights: &metal::Buffer,
    bias: &metal::Buffer,
    cfg: MetalReadoutSgdConfig,
) -> Result<()> {
    let cfg_buf = shared_bytes(device, &cfg);
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (idx, buffer) in [features, targets, weights, bias, &cfg_buf]
        .iter()
        .enumerate()
    {
        encoder.set_buffer(idx as u64, Some(buffer), 0);
    }
    let threads = pipeline.max_total_threads_per_threadgroup().min(256).max(1);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_readout_bag_sgd_chunk(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    features: &metal::Buffer,
    target_bags: &metal::Buffer,
    weights: &metal::Buffer,
    bias: &metal::Buffer,
    cfg: MetalReadoutBagSgdConfig,
) -> Result<()> {
    let cfg_buf = shared_bytes(device, &cfg);
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (idx, buffer) in [features, target_bags, weights, bias, &cfg_buf]
        .iter()
        .enumerate()
    {
        encoder.set_buffer(idx as u64, Some(buffer), 0);
    }
    let threads = pipeline.max_total_threads_per_threadgroup().min(256).max(1);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_state_linear_update(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    weights: &metal::Buffer,
    bias: &metal::Buffer,
    input: &metal::Buffer,
    output: &metal::Buffer,
    target: &metal::Buffer,
    cfg: &metal::Buffer,
    total_weights: usize,
) -> Result<()> {
    if total_weights == 0 {
        return Ok(());
    }
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (idx, buffer) in [weights, bias, input, output, target, cfg]
        .iter()
        .enumerate()
    {
        encoder.set_buffer(idx as u64, Some(buffer), 0);
    }
    let threads = pipeline.max_total_threads_per_threadgroup().min(256).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: total_weights as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

fn encode_state_hv_projection_update(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    weights: &metal::Buffer,
    reservoir_state: &metal::Buffer,
    target_hv: &metal::Buffer,
    cfg: &metal::Buffer,
    total_weights: usize,
) -> Result<()> {
    if total_weights == 0 {
        return Ok(());
    }
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (idx, buffer) in [weights, reservoir_state, target_hv, cfg]
        .iter()
        .enumerate()
    {
        encoder.set_buffer(idx as u64, Some(buffer), 0);
    }
    let threads = pipeline.max_total_threads_per_threadgroup().min(256).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: total_weights as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

fn shared_buffer_with_data(device: &Device, values: &[f32]) -> metal::Buffer {
    device.new_buffer_with_data(
        values.as_ptr().cast::<c_void>(),
        std::mem::size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn shared_i32_buffer_with_data(device: &Device, values: &[i32]) -> metal::Buffer {
    device.new_buffer_with_data(
        values.as_ptr().cast::<c_void>(),
        std::mem::size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn shared_zero_buffer(device: &Device, len: usize) -> metal::Buffer {
    device.new_buffer(
        (len * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn shared_bytes<T>(device: &Device, value: &T) -> metal::Buffer {
    device.new_buffer_with_data(
        (value as *const T).cast::<c_void>(),
        std::mem::size_of::<T>() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn read_f32_buffer(buffer: &metal::Buffer, len: usize) -> Vec<f32> {
    let ptr = buffer.contents().cast::<f32>();
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
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
    if observations.rows % config.batch != 0 {
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
    expected_len(
        &params.obs_to_h,
        config.hidden_dim * config.obs_dim,
        "obs_to_h",
    )?;
    expected_len(
        &params.h_recurrent,
        config.hidden_dim * config.hidden_dim,
        "h_recurrent",
    )?;
    expected_len(&params.h_bias, config.hidden_dim, "h_bias")?;
    expected_len(
        &params.reservoir_input,
        config.reservoir_dim * config.hidden_dim,
        "reservoir_input",
    )?;
    expected_len(
        &params.reservoir_recurrent,
        config.reservoir_dim * config.reservoir_dim,
        "reservoir_recurrent",
    )?;
    expected_len(
        &params.reservoir_bias,
        config.reservoir_dim,
        "reservoir_bias",
    )?;
    expected_len(
        &params.hv_proj,
        config.hv_dim * config.reservoir_dim,
        "hv_proj",
    )?;
    Ok(())
}

fn expected_len(values: &[f32], expected: usize, label: &str) -> Result<()> {
    if values.len() != expected {
        return Err(SymError::Shape(format!(
            "{label} expected {expected} values, got {}",
            values.len()
        )));
    }
    Ok(())
}
