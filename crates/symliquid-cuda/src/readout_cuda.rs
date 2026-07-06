#[cfg(feature = "cuda")]
use std::collections::BTreeMap;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use rand::SeedableRng;
use symliquid_core::error::{Result, SymError};
use symliquid_core::tensor::Tensor;
use symliquid_core::token_superposition::TokenSuperpositionConfig;
#[cfg(feature = "cuda")]
use symliquid_core::token_superposition::{
    build_token_superposition_dataset, build_token_superposition_report, evaluate_token_readout,
    make_ar_training_batch, make_bag_training_batch, token_feature_table,
    TokenSuperpositionRunReport,
};
use symliquid_core::train::{LinearReadout, TrainingTrace};

#[cfg(feature = "cuda")]
const READOUT_KERNELS: &str = include_str!("../kernels/readout_kernels.cu");

#[cfg(feature = "cuda")]
static LINEAR_READOUT_TOPK_SESSION_CREATES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static LINEAR_READOUT_TOPK_SESSION_REUSES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static LINEAR_READOUT_TOPK_SESSION_EVICTIONS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static LINEAR_READOUT_TOPK_SESSION_RESIZES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static LINEAR_READOUT_TOPK_SESSION_PREPARES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static LINEAR_READOUT_TOPK_SESSION_MAX_ENTRIES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static LINEAR_READOUT_TOPK_CALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static LINEAR_READOUT_TOPK_ROWS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static WEIGHTED_SCORE_SESSION_CREATES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static WEIGHTED_SCORE_SESSION_REUSES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static WEIGHTED_SCORE_SESSION_EVICTIONS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static WEIGHTED_SCORE_SESSION_RESIZES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static WEIGHTED_SCORE_SESSION_PREPARES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static WEIGHTED_SCORE_SESSION_MAX_ENTRIES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static WEIGHTED_SCORE_CALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "cuda")]
static WEIGHTED_SCORE_ROWS: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaLinearReadoutRuntimeSummary {
    pub topk_session_create_count: usize,
    pub topk_session_reuse_count: usize,
    pub topk_session_eviction_count: usize,
    pub topk_session_resize_count: usize,
    pub topk_session_prepare_count: usize,
    pub topk_session_max_entry_count: usize,
    pub topk_thread_session_count: usize,
    pub topk_call_count: usize,
    pub topk_row_count: usize,
    pub weighted_score_session_create_count: usize,
    pub weighted_score_session_reuse_count: usize,
    pub weighted_score_session_eviction_count: usize,
    pub weighted_score_session_resize_count: usize,
    pub weighted_score_session_prepare_count: usize,
    pub weighted_score_session_max_entry_count: usize,
    pub weighted_score_thread_session_count: usize,
    pub weighted_score_call_count: usize,
    pub weighted_score_row_count: usize,
}

#[cfg(feature = "cuda")]
struct ReadoutKernelCache {
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    function: cudarc::driver::CudaFunction,
    bag_function: cudarc::driver::CudaFunction,
    code_fast_function: cudarc::driver::CudaFunction,
    binary_score_function: cudarc::driver::CudaFunction,
    weighted_score_function: cudarc::driver::CudaFunction,
    logits_function: cudarc::driver::CudaFunction,
    topk_function: cudarc::driver::CudaFunction,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReadoutDeviceKey {
    weights_ptr: usize,
    bias_ptr: usize,
    weights_len: usize,
    bias_len: usize,
    input_dim: usize,
    output_dim: usize,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WeightedScoreDeviceKey {
    weights_ptr: usize,
    weights_len: usize,
    feature_dim: usize,
}

#[cfg(feature = "cuda")]
impl WeightedScoreDeviceKey {
    fn from_weights(weights: &[f32]) -> Self {
        Self {
            weights_ptr: weights.as_ptr() as usize,
            weights_len: weights.len(),
            feature_dim: weights.len(),
        }
    }
}

#[cfg(feature = "cuda")]
impl ReadoutDeviceKey {
    fn from_readout(readout: &LinearReadout) -> Self {
        Self {
            weights_ptr: readout.weights.as_ptr() as usize,
            bias_ptr: readout.bias.as_ptr() as usize,
            weights_len: readout.weights.len(),
            bias_len: readout.bias.len(),
            input_dim: readout.input_dim,
            output_dim: readout.output_dim,
        }
    }
}

#[cfg(feature = "cuda")]
pub struct CudaLinearReadoutSession {
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    key: ReadoutDeviceKey,
    weights_dev: cudarc::driver::CudaSlice<f32>,
    bias_dev: cudarc::driver::CudaSlice<f32>,
    features_dev: cudarc::driver::CudaSlice<f32>,
    logits_dev: cudarc::driver::CudaSlice<f32>,
    indices_dev: cudarc::driver::CudaSlice<i32>,
    log_probs_dev: cudarc::driver::CudaSlice<f32>,
    row_capacity: usize,
    k_capacity: usize,
}

#[cfg(feature = "cuda")]
pub struct CudaWeightedFeatureScoreSession {
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    key: WeightedScoreDeviceKey,
    weights_dev: cudarc::driver::CudaSlice<f32>,
    features_dev: cudarc::driver::CudaSlice<f32>,
    scores_dev: cudarc::driver::CudaSlice<f32>,
    row_capacity: usize,
}

#[cfg(feature = "cuda")]
impl CudaLinearReadoutSession {
    pub fn new(readout: &LinearReadout, row_capacity: usize, k_capacity: usize) -> Result<Self> {
        let kernel = readout_kernel_cache()?;
        let stream = kernel.stream.clone();
        let row_capacity = row_capacity.max(1);
        let k_capacity = k_capacity.max(1).min(readout.output_dim.max(1));
        Ok(Self {
            stream: stream.clone(),
            key: ReadoutDeviceKey::from_readout(readout),
            weights_dev: stream.clone_htod(&readout.weights).map_err(cuda_error)?,
            bias_dev: stream.clone_htod(&readout.bias).map_err(cuda_error)?,
            features_dev: stream
                .clone_htod(&vec![0.0f32; row_capacity * readout.input_dim])
                .map_err(cuda_error)?,
            logits_dev: stream
                .clone_htod(&vec![0.0f32; row_capacity * readout.output_dim])
                .map_err(cuda_error)?,
            indices_dev: stream
                .clone_htod(&vec![0i32; row_capacity * k_capacity])
                .map_err(cuda_error)?,
            log_probs_dev: stream
                .clone_htod(&vec![0.0f32; row_capacity * k_capacity])
                .map_err(cuda_error)?,
            row_capacity,
            k_capacity,
        })
    }

    pub fn matches_readout(&self, readout: &LinearReadout) -> bool {
        self.key == ReadoutDeviceKey::from_readout(readout)
    }

    fn ensure_capacity(&mut self, readout: &LinearReadout, rows: usize, k: usize) -> Result<bool> {
        let rows = rows.max(1);
        let k = k.max(1).min(readout.output_dim.max(1));
        let mut resized = false;
        if !self.matches_readout(readout) {
            self.key = ReadoutDeviceKey::from_readout(readout);
            self.weights_dev = self
                .stream
                .clone_htod(&readout.weights)
                .map_err(cuda_error)?;
            self.bias_dev = self.stream.clone_htod(&readout.bias).map_err(cuda_error)?;
            self.row_capacity = 0;
            self.k_capacity = 0;
            resized = true;
        }
        if self.row_capacity < rows || self.k_capacity < k {
            self.row_capacity = self.row_capacity.max(rows).next_power_of_two();
            self.k_capacity = self
                .k_capacity
                .max(k)
                .next_power_of_two()
                .min(readout.output_dim.max(1));
            self.features_dev = self
                .stream
                .clone_htod(&vec![0.0f32; self.row_capacity * readout.input_dim])
                .map_err(cuda_error)?;
            self.logits_dev = self
                .stream
                .clone_htod(&vec![0.0f32; self.row_capacity * readout.output_dim])
                .map_err(cuda_error)?;
            self.indices_dev = self
                .stream
                .clone_htod(&vec![0i32; self.row_capacity * self.k_capacity])
                .map_err(cuda_error)?;
            self.log_probs_dev = self
                .stream
                .clone_htod(&vec![0.0f32; self.row_capacity * self.k_capacity])
                .map_err(cuda_error)?;
            resized = true;
        }
        Ok(resized)
    }
}

#[cfg(feature = "cuda")]
impl CudaWeightedFeatureScoreSession {
    pub fn new(weights: &[f32], row_capacity: usize) -> Result<Self> {
        if weights.is_empty() {
            return Err(SymError::InvalidArgument(
                "CUDA weighted feature scorer requires at least one weight".to_string(),
            ));
        }
        let kernel = readout_kernel_cache()?;
        let stream = kernel.stream.clone();
        let row_capacity = row_capacity.max(1);
        Ok(Self {
            stream: stream.clone(),
            key: WeightedScoreDeviceKey::from_weights(weights),
            weights_dev: stream.clone_htod(weights).map_err(cuda_error)?,
            features_dev: stream
                .clone_htod(&vec![0.0f32; row_capacity * weights.len()])
                .map_err(cuda_error)?,
            scores_dev: stream
                .clone_htod(&vec![0.0f32; row_capacity])
                .map_err(cuda_error)?,
            row_capacity,
        })
    }

    fn matches_weights(&self, weights: &[f32]) -> bool {
        self.key == WeightedScoreDeviceKey::from_weights(weights)
    }

    fn ensure_capacity(&mut self, weights: &[f32], rows: usize) -> Result<bool> {
        if weights.is_empty() {
            return Err(SymError::InvalidArgument(
                "CUDA weighted feature scorer requires at least one weight".to_string(),
            ));
        }
        let rows = rows.max(1);
        let mut resized = false;
        if !self.matches_weights(weights) {
            self.key = WeightedScoreDeviceKey::from_weights(weights);
            self.weights_dev = self.stream.clone_htod(weights).map_err(cuda_error)?;
            self.row_capacity = 0;
            resized = true;
        }
        if self.row_capacity < rows {
            self.row_capacity = self.row_capacity.max(rows).next_power_of_two();
            self.features_dev = self
                .stream
                .clone_htod(&vec![0.0f32; self.row_capacity * weights.len()])
                .map_err(cuda_error)?;
            self.scores_dev = self
                .stream
                .clone_htod(&vec![0.0f32; self.row_capacity])
                .map_err(cuda_error)?;
            resized = true;
        }
        Ok(resized)
    }
}

#[cfg(feature = "cuda")]
thread_local! {
    static THREAD_READOUT_SESSIONS: std::cell::RefCell<Vec<CudaLinearReadoutSession>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static THREAD_WEIGHTED_SCORE_SESSIONS: std::cell::RefCell<Vec<CudaWeightedFeatureScoreSession>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(feature = "cuda")]
pub fn clear_thread_readout_session() {
    THREAD_READOUT_SESSIONS.with(|cell| {
        cell.borrow_mut().clear();
    });
    THREAD_WEIGHTED_SCORE_SESSIONS.with(|cell| {
        cell.borrow_mut().clear();
    });
}

#[cfg(feature = "cuda")]
fn readout_session_cache_limit() -> usize {
    std::env::var("THESEUS_CUDA_READOUT_SESSION_CACHE_LIMIT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(4)
        .clamp(1, 16)
}

#[cfg(feature = "cuda")]
fn weighted_score_session_cache_limit() -> usize {
    std::env::var("THESEUS_CUDA_WEIGHTED_SCORE_SESSION_CACHE_LIMIT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(4)
        .clamp(1, 16)
}

#[cfg(feature = "cuda")]
fn resident_readout_session_index(
    sessions: &mut Vec<CudaLinearReadoutSession>,
    readout: &LinearReadout,
    row_capacity: usize,
    k_capacity: usize,
) -> Result<(usize, bool)> {
    if let Some(index) = sessions
        .iter()
        .position(|session| session.matches_readout(readout))
    {
        return Ok((index, false));
    }
    let limit = readout_session_cache_limit();
    while sessions.len() >= limit {
        sessions.remove(0);
        LINEAR_READOUT_TOPK_SESSION_EVICTIONS.fetch_add(1, Ordering::Relaxed);
    }
    sessions.push(CudaLinearReadoutSession::new(
        readout,
        row_capacity.max(1),
        k_capacity.max(1),
    )?);
    LINEAR_READOUT_TOPK_SESSION_CREATES.fetch_add(1, Ordering::Relaxed);
    LINEAR_READOUT_TOPK_SESSION_MAX_ENTRIES.fetch_max(sessions.len(), Ordering::Relaxed);
    Ok((sessions.len() - 1, true))
}

#[cfg(feature = "cuda")]
fn resident_weighted_score_session_index(
    sessions: &mut Vec<CudaWeightedFeatureScoreSession>,
    weights: &[f32],
    row_capacity: usize,
) -> Result<(usize, bool)> {
    if let Some(index) = sessions
        .iter()
        .position(|session| session.matches_weights(weights))
    {
        return Ok((index, false));
    }
    let limit = weighted_score_session_cache_limit();
    while sessions.len() >= limit {
        sessions.remove(0);
        WEIGHTED_SCORE_SESSION_EVICTIONS.fetch_add(1, Ordering::Relaxed);
    }
    sessions.push(CudaWeightedFeatureScoreSession::new(
        weights,
        row_capacity.max(1),
    )?);
    WEIGHTED_SCORE_SESSION_CREATES.fetch_add(1, Ordering::Relaxed);
    WEIGHTED_SCORE_SESSION_MAX_ENTRIES.fetch_max(sessions.len(), Ordering::Relaxed);
    Ok((sessions.len() - 1, true))
}

#[cfg(feature = "cuda")]
pub fn prepare_thread_readout_session_cuda(
    readout: &LinearReadout,
    row_capacity: usize,
    k_capacity: usize,
) -> Result<()> {
    if readout.input_dim == 0 || readout.output_dim == 0 || row_capacity == 0 || k_capacity == 0 {
        return Ok(());
    }
    THREAD_READOUT_SESSIONS.with(|cell| {
        let mut sessions = cell.borrow_mut();
        let (index, _created) =
            resident_readout_session_index(&mut sessions, readout, row_capacity, k_capacity)?;
        if sessions[index].ensure_capacity(readout, row_capacity, k_capacity)? {
            LINEAR_READOUT_TOPK_SESSION_RESIZES.fetch_add(1, Ordering::Relaxed);
        }
        LINEAR_READOUT_TOPK_SESSION_PREPARES.fetch_add(1, Ordering::Relaxed);
        LINEAR_READOUT_TOPK_SESSION_MAX_ENTRIES.fetch_max(sessions.len(), Ordering::Relaxed);
        Ok(())
    })
}

#[cfg(feature = "cuda")]
pub fn prepare_thread_weighted_feature_score_session_cuda(
    weights: &[f32],
    row_capacity: usize,
) -> Result<()> {
    if weights.is_empty() {
        return Err(SymError::InvalidArgument(
            "CUDA weighted feature scorer requires at least one weight".to_string(),
        ));
    }
    THREAD_WEIGHTED_SCORE_SESSIONS.with(|cell| {
        let mut sessions = cell.borrow_mut();
        let (index, _created) =
            resident_weighted_score_session_index(&mut sessions, weights, row_capacity)?;
        if sessions[index].ensure_capacity(weights, row_capacity)? {
            WEIGHTED_SCORE_SESSION_RESIZES.fetch_add(1, Ordering::Relaxed);
        }
        WEIGHTED_SCORE_SESSION_PREPARES.fetch_add(1, Ordering::Relaxed);
        WEIGHTED_SCORE_SESSION_MAX_ENTRIES.fetch_max(sessions.len(), Ordering::Relaxed);
        Ok(())
    })
}

#[cfg(feature = "cuda")]
pub fn thread_readout_session_count_cuda() -> usize {
    THREAD_READOUT_SESSIONS.with(|cell| cell.borrow().len())
}

#[cfg(feature = "cuda")]
pub fn thread_weighted_score_session_count_cuda() -> usize {
    THREAD_WEIGHTED_SCORE_SESSIONS.with(|cell| cell.borrow().len())
}

#[cfg(feature = "cuda")]
pub fn linear_readout_runtime_summary_cuda() -> CudaLinearReadoutRuntimeSummary {
    let thread_session_count = thread_readout_session_count_cuda();
    let weighted_thread_session_count = thread_weighted_score_session_count_cuda();
    CudaLinearReadoutRuntimeSummary {
        topk_session_create_count: LINEAR_READOUT_TOPK_SESSION_CREATES.load(Ordering::Relaxed),
        topk_session_reuse_count: LINEAR_READOUT_TOPK_SESSION_REUSES.load(Ordering::Relaxed),
        topk_session_eviction_count: LINEAR_READOUT_TOPK_SESSION_EVICTIONS.load(Ordering::Relaxed),
        topk_session_resize_count: LINEAR_READOUT_TOPK_SESSION_RESIZES.load(Ordering::Relaxed),
        topk_session_prepare_count: LINEAR_READOUT_TOPK_SESSION_PREPARES.load(Ordering::Relaxed),
        topk_session_max_entry_count: LINEAR_READOUT_TOPK_SESSION_MAX_ENTRIES
            .load(Ordering::Relaxed)
            .max(thread_session_count),
        topk_thread_session_count: thread_session_count,
        topk_call_count: LINEAR_READOUT_TOPK_CALLS.load(Ordering::Relaxed),
        topk_row_count: LINEAR_READOUT_TOPK_ROWS.load(Ordering::Relaxed),
        weighted_score_session_create_count: WEIGHTED_SCORE_SESSION_CREATES.load(Ordering::Relaxed),
        weighted_score_session_reuse_count: WEIGHTED_SCORE_SESSION_REUSES.load(Ordering::Relaxed),
        weighted_score_session_eviction_count: WEIGHTED_SCORE_SESSION_EVICTIONS
            .load(Ordering::Relaxed),
        weighted_score_session_resize_count: WEIGHTED_SCORE_SESSION_RESIZES.load(Ordering::Relaxed),
        weighted_score_session_prepare_count: WEIGHTED_SCORE_SESSION_PREPARES
            .load(Ordering::Relaxed),
        weighted_score_session_max_entry_count: WEIGHTED_SCORE_SESSION_MAX_ENTRIES
            .load(Ordering::Relaxed)
            .max(weighted_thread_session_count),
        weighted_score_thread_session_count: weighted_thread_session_count,
        weighted_score_call_count: WEIGHTED_SCORE_CALLS.load(Ordering::Relaxed),
        weighted_score_row_count: WEIGHTED_SCORE_ROWS.load(Ordering::Relaxed),
    }
}

#[cfg(feature = "cuda")]
static READOUT_KERNEL_CACHE: std::sync::OnceLock<std::result::Result<ReadoutKernelCache, String>> =
    std::sync::OnceLock::new();

#[cfg(feature = "cuda")]
fn readout_kernel_cache() -> Result<&'static ReadoutKernelCache> {
    match READOUT_KERNEL_CACHE.get_or_init(init_readout_kernel_cache) {
        Ok(cache) => Ok(cache),
        Err(error) => Err(SymError::InvalidArgument(format!(
            "CUDA readout kernel cache initialization failed: {error}"
        ))),
    }
}

#[cfg(feature = "cuda")]
fn init_readout_kernel_cache() -> std::result::Result<ReadoutKernelCache, String> {
    use cudarc::driver::CudaContext;
    use cudarc::nvrtc::compile_ptx;

    let ctx = CudaContext::new(0).map_err(|error| error.to_string())?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(READOUT_KERNELS).map_err(|error| error.to_string())?;
    let module = ctx.load_module(ptx).map_err(|error| error.to_string())?;
    let function = module
        .load_function("readout_sgd_samples_kernel")
        .map_err(|error| error.to_string())?;
    let bag_function = module
        .load_function("readout_bag_sgd_samples_kernel")
        .map_err(|error| error.to_string())?;
    let code_fast_function = module
        .load_function("code_fast_readout_train_kernel")
        .map_err(|error| error.to_string())?;
    let binary_score_function = module
        .load_function("binary_readout_score_kernel")
        .map_err(|error| error.to_string())?;
    let weighted_score_function = module
        .load_function("weighted_feature_score_kernel")
        .map_err(|error| error.to_string())?;
    let logits_function = module
        .load_function("linear_readout_logits_kernel")
        .map_err(|error| error.to_string())?;
    let topk_function = module
        .load_function("topk_log_probs_kernel")
        .map_err(|error| error.to_string())?;
    Ok(ReadoutKernelCache {
        stream,
        function,
        bag_function,
        code_fast_function,
        binary_score_function,
        weighted_score_function,
        logits_function,
        topk_function,
    })
}

#[cfg(feature = "cuda")]
pub fn train_readout_sgd_cuda(
    features: &Tensor,
    targets: &[usize],
    readout: &mut LinearReadout,
    epochs: usize,
    lr: f32,
    samples_per_launch: usize,
) -> Result<TrainingTrace> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    features.ensure_cols(readout.input_dim, "CUDA readout features")?;
    if targets.len() != features.rows {
        return Err(SymError::Shape(format!(
            "target batch expected {}, got {}",
            features.rows,
            targets.len()
        )));
    }
    if readout.output_dim > 256 {
        return Err(SymError::InvalidArgument(format!(
            "CUDA readout trainer supports output_dim <= 256, got {}",
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

    let targets_i32 = targets
        .iter()
        .map(|target| *target as i32)
        .collect::<Vec<_>>();
    let chunk = samples_per_launch.clamp(1, features.rows.max(1));

    let kernel = readout_kernel_cache()?;
    let stream = &kernel.stream;
    let function = &kernel.function;

    let features_dev = stream.clone_htod(&features.data).map_err(cuda_error)?;
    let targets_dev = stream.clone_htod(&targets_i32).map_err(cuda_error)?;
    let mut weights_dev = stream.clone_htod(&readout.weights).map_err(cuda_error)?;
    let mut bias_dev = stream.clone_htod(&readout.bias).map_err(cuda_error)?;
    let input_dim = readout.input_dim as i32;
    let output_dim = readout.output_dim as i32;
    let cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    for _ in 0..epochs {
        for sample_start in (0..features.rows).step_by(chunk) {
            let sample_count = (features.rows - sample_start).min(chunk) as i32;
            let sample_start = sample_start as i32;
            let mut args = stream.launch_builder(function);
            args.arg(&features_dev);
            args.arg(&targets_dev);
            args.arg(&mut weights_dev);
            args.arg(&mut bias_dev);
            args.arg(&sample_start);
            args.arg(&sample_count);
            args.arg(&input_dim);
            args.arg(&output_dim);
            args.arg(&lr);
            unsafe {
                args.launch(cfg).map_err(cuda_error)?;
            }
        }
    }

    readout.weights = stream.clone_dtoh(&weights_dev).map_err(cuda_error)?;
    readout.bias = stream.clone_dtoh(&bias_dev).map_err(cuda_error)?;
    Ok(TrainingTrace {
        loss: 0.0,
        accuracy: 0.0,
        grad_norm: 0.0,
    })
}

#[cfg(feature = "cuda")]
pub fn train_readout_bag_sgd_cuda(
    features: &Tensor,
    target_bags: &[usize],
    targets_per_sample: usize,
    readout: &mut LinearReadout,
    epochs: usize,
    lr: f32,
    samples_per_launch: usize,
) -> Result<TrainingTrace> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    features.ensure_cols(readout.input_dim, "CUDA bag readout features")?;
    if targets_per_sample == 0 {
        return Err(SymError::InvalidArgument(
            "targets_per_sample must be greater than zero".to_string(),
        ));
    }
    if target_bags.len() != features.rows * targets_per_sample {
        return Err(SymError::Shape(format!(
            "target bag batch expected {}, got {}",
            features.rows * targets_per_sample,
            target_bags.len()
        )));
    }
    if readout.output_dim > 256 {
        return Err(SymError::InvalidArgument(format!(
            "CUDA bag readout trainer supports output_dim <= 256, got {}",
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

    let target_bags_i32 = target_bags
        .iter()
        .map(|target| *target as i32)
        .collect::<Vec<_>>();
    let chunk = samples_per_launch.clamp(1, features.rows.max(1));

    let kernel = readout_kernel_cache()?;
    let stream = &kernel.stream;
    let function = &kernel.bag_function;

    let features_dev = stream.clone_htod(&features.data).map_err(cuda_error)?;
    let targets_dev = stream.clone_htod(&target_bags_i32).map_err(cuda_error)?;
    let mut weights_dev = stream.clone_htod(&readout.weights).map_err(cuda_error)?;
    let mut bias_dev = stream.clone_htod(&readout.bias).map_err(cuda_error)?;
    let input_dim = readout.input_dim as i32;
    let output_dim = readout.output_dim as i32;
    let targets_per_sample = targets_per_sample as i32;
    let cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    for _ in 0..epochs {
        for sample_start in (0..features.rows).step_by(chunk) {
            let sample_count = (features.rows - sample_start).min(chunk) as i32;
            let sample_start = sample_start as i32;
            let mut args = stream.launch_builder(function);
            args.arg(&features_dev);
            args.arg(&targets_dev);
            args.arg(&mut weights_dev);
            args.arg(&mut bias_dev);
            args.arg(&sample_start);
            args.arg(&sample_count);
            args.arg(&input_dim);
            args.arg(&output_dim);
            args.arg(&targets_per_sample);
            args.arg(&lr);
            unsafe {
                args.launch(cfg).map_err(cuda_error)?;
            }
        }
    }

    readout.weights = stream.clone_dtoh(&weights_dev).map_err(cuda_error)?;
    readout.bias = stream.clone_dtoh(&bias_dev).map_err(cuda_error)?;
    readout.train_batch_target_bags(features, target_bags, targets_per_sample as usize, 0.0)
}

#[cfg(feature = "cuda")]
pub fn train_code_fast_readout_cuda(
    features: &Tensor,
    targets: &[usize],
    salts: &[usize],
    readout: &mut LinearReadout,
    epochs: usize,
    lr: f32,
) -> Result<TrainingTrace> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    features.ensure_cols(readout.input_dim, "CUDA Code LM fast readout features")?;
    if targets.len() != features.rows || salts.len() != features.rows {
        return Err(SymError::Shape(format!(
            "targets/salts expected {} rows, got targets={} salts={}",
            features.rows,
            targets.len(),
            salts.len()
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

    let targets_i32 = targets
        .iter()
        .map(|target| *target as i32)
        .collect::<Vec<_>>();
    let salts_i32 = salts.iter().map(|salt| *salt as i32).collect::<Vec<_>>();
    let kernel = readout_kernel_cache()?;
    let stream = &kernel.stream;
    let function = &kernel.code_fast_function;

    let features_dev = stream.clone_htod(&features.data).map_err(cuda_error)?;
    let targets_dev = stream.clone_htod(&targets_i32).map_err(cuda_error)?;
    let salts_dev = stream.clone_htod(&salts_i32).map_err(cuda_error)?;
    let mut weights_dev = stream.clone_htod(&readout.weights).map_err(cuda_error)?;
    let mut bias_dev = stream.clone_htod(&readout.bias).map_err(cuda_error)?;
    let input_dim = readout.input_dim as i32;
    let output_dim = readout.output_dim as i32;
    let block_x = 256u32;
    let grid_x = ((readout.input_dim as u32) + block_x - 1) / block_x;
    let samples_per_launch = 65_535usize.min(features.rows.max(1));

    for _ in 0..epochs.max(1) {
        for sample_start in (0..features.rows).step_by(samples_per_launch) {
            let sample_count = (features.rows - sample_start).min(samples_per_launch) as i32;
            let sample_start = sample_start as i32;
            let cfg = LaunchConfig {
                block_dim: (block_x, 1, 1),
                grid_dim: (grid_x.max(1), sample_count.max(1) as u32, 1),
                shared_mem_bytes: 0,
            };
            let mut args = stream.launch_builder(function);
            args.arg(&features_dev);
            args.arg(&targets_dev);
            args.arg(&salts_dev);
            args.arg(&mut weights_dev);
            args.arg(&mut bias_dev);
            args.arg(&sample_start);
            args.arg(&sample_count);
            args.arg(&input_dim);
            args.arg(&output_dim);
            args.arg(&lr);
            unsafe {
                args.launch(cfg).map_err(cuda_error)?;
            }
        }
    }

    readout.weights = stream.clone_dtoh(&weights_dev).map_err(cuda_error)?;
    readout.bias = stream.clone_dtoh(&bias_dev).map_err(cuda_error)?;
    readout.evaluate_batch(features, targets)
}

#[cfg(feature = "cuda")]
pub fn score_binary_readout_cuda(features: &Tensor, readout: &LinearReadout) -> Result<Vec<f32>> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    features.ensure_cols(readout.input_dim, "CUDA binary readout scoring features")?;
    if readout.output_dim < 2 {
        return Err(SymError::InvalidArgument(format!(
            "binary readout scoring requires output_dim >= 2, got {}",
            readout.output_dim
        )));
    }
    if features.rows == 0 {
        return Ok(Vec::new());
    }

    let kernel = readout_kernel_cache()?;
    let stream = &kernel.stream;
    let function = &kernel.binary_score_function;

    let features_dev = stream.clone_htod(&features.data).map_err(cuda_error)?;
    let weights_dev = stream.clone_htod(&readout.weights).map_err(cuda_error)?;
    let bias_dev = stream.clone_htod(&readout.bias).map_err(cuda_error)?;
    let mut scores_dev = stream
        .clone_htod(&vec![0.0f32; features.rows])
        .map_err(cuda_error)?;
    let sample_count = features.rows as i32;
    let input_dim = readout.input_dim as i32;
    let output_dim = readout.output_dim as i32;
    let cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (features.rows as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut args = stream.launch_builder(function);
    args.arg(&features_dev);
    args.arg(&weights_dev);
    args.arg(&bias_dev);
    args.arg(&mut scores_dev);
    args.arg(&sample_count);
    args.arg(&input_dim);
    args.arg(&output_dim);
    unsafe {
        args.launch(cfg).map_err(cuda_error)?;
    }

    stream.clone_dtoh(&scores_dev).map_err(cuda_error)
}

#[cfg(feature = "cuda")]
pub fn weighted_feature_scores_cuda(
    features: &Tensor,
    weights: &[f32],
    bias: f32,
) -> Result<Vec<f32>> {
    if features.cols != weights.len() {
        return Err(SymError::Shape(format!(
            "weighted CUDA scorer expected {} weights for {} feature columns",
            features.cols,
            weights.len()
        )));
    }
    if features.rows == 0 {
        return Ok(Vec::new());
    }

    THREAD_WEIGHTED_SCORE_SESSIONS.with(|cell| {
        let mut sessions = cell.borrow_mut();
        let (index, created) =
            resident_weighted_score_session_index(&mut sessions, weights, features.rows)?;
        if !created {
            WEIGHTED_SCORE_SESSION_REUSES.fetch_add(1, Ordering::Relaxed);
        }
        WEIGHTED_SCORE_CALLS.fetch_add(1, Ordering::Relaxed);
        WEIGHTED_SCORE_ROWS.fetch_add(features.rows, Ordering::Relaxed);
        WEIGHTED_SCORE_SESSION_MAX_ENTRIES.fetch_max(sessions.len(), Ordering::Relaxed);
        let session = sessions.get_mut(index).ok_or_else(|| {
            SymError::InvalidArgument("CUDA weighted feature score session missing".to_string())
        })?;
        weighted_feature_scores_cuda_session(session, features, weights, bias)
    })
}

#[cfg(feature = "cuda")]
pub fn weighted_feature_scores_cuda_session(
    session: &mut CudaWeightedFeatureScoreSession,
    features: &Tensor,
    weights: &[f32],
    bias: f32,
) -> Result<Vec<f32>> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    if features.cols != weights.len() {
        return Err(SymError::Shape(format!(
            "weighted CUDA scorer session expected {} weights for {} feature columns",
            features.cols,
            weights.len()
        )));
    }
    if features.rows == 0 {
        return Ok(Vec::new());
    }
    if session.ensure_capacity(weights, features.rows)? {
        WEIGHTED_SCORE_SESSION_RESIZES.fetch_add(1, Ordering::Relaxed);
    }
    if session.features_dev.len() < features.data.len() || session.scores_dev.len() < features.rows
    {
        session.row_capacity = 0;
        if session.ensure_capacity(weights, features.rows)? {
            WEIGHTED_SCORE_SESSION_RESIZES.fetch_add(1, Ordering::Relaxed);
        }
    }
    if session.features_dev.len() < features.data.len() || session.scores_dev.len() < features.rows
    {
        return Err(SymError::Shape(format!(
            "CUDA weighted score session capacity shortfall: features_dev={} required_features={} scores_dev={} required_scores={}",
            session.features_dev.len(),
            features.data.len(),
            session.scores_dev.len(),
            features.rows
        )));
    }
    session
        .stream
        .memcpy_htod(&features.data, &mut session.features_dev)
        .map_err(cuda_error)?;

    let kernel = readout_kernel_cache()?;
    let stream = &session.stream;
    let function = &kernel.weighted_score_function;

    let sample_count = features.rows as i32;
    let feature_dim = features.cols as i32;
    let cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (features.rows as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut args = stream.launch_builder(function);
    args.arg(&session.features_dev);
    args.arg(&session.weights_dev);
    args.arg(&bias);
    args.arg(&mut session.scores_dev);
    args.arg(&sample_count);
    args.arg(&feature_dim);
    unsafe {
        args.launch(cfg).map_err(cuda_error)?;
    }

    let mut scores = stream.clone_dtoh(&session.scores_dev).map_err(cuda_error)?;
    scores.truncate(features.rows);
    Ok(scores)
}

#[cfg(feature = "cuda")]
pub fn linear_readout_logits_cuda(features: &Tensor, readout: &LinearReadout) -> Result<Tensor> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    features.ensure_cols(readout.input_dim, "CUDA linear readout logits features")?;
    if features.rows == 0 {
        return Tensor::new(0, readout.output_dim, Vec::new());
    }

    let kernel = readout_kernel_cache()?;
    let stream = &kernel.stream;
    let function = &kernel.logits_function;

    let features_dev = stream.clone_htod(&features.data).map_err(cuda_error)?;
    let weights_dev = stream.clone_htod(&readout.weights).map_err(cuda_error)?;
    let bias_dev = stream.clone_htod(&readout.bias).map_err(cuda_error)?;
    let mut logits_dev = stream
        .clone_htod(&vec![0.0f32; features.rows * readout.output_dim])
        .map_err(cuda_error)?;
    let sample_count = features.rows as i32;
    let input_dim = readout.input_dim as i32;
    let output_dim = readout.output_dim as i32;
    let cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (features.rows as u32, readout.output_dim as u32, 1),
        shared_mem_bytes: 0,
    };

    let mut args = stream.launch_builder(function);
    args.arg(&features_dev);
    args.arg(&weights_dev);
    args.arg(&bias_dev);
    args.arg(&mut logits_dev);
    args.arg(&sample_count);
    args.arg(&input_dim);
    args.arg(&output_dim);
    unsafe {
        args.launch(cfg).map_err(cuda_error)?;
    }

    let data = stream.clone_dtoh(&logits_dev).map_err(cuda_error)?;
    Tensor::new(features.rows, readout.output_dim, data)
}

#[cfg(feature = "cuda")]
pub fn linear_readout_topk_log_probs_cuda(
    features: &Tensor,
    readout: &LinearReadout,
    limit: usize,
    top_p: f32,
) -> Result<Vec<Vec<(usize, f32)>>> {
    features.ensure_cols(readout.input_dim, "CUDA linear readout top-k features")?;
    if features.rows == 0 || readout.output_dim == 0 || limit == 0 {
        return Ok(vec![Vec::new(); features.rows]);
    }
    let k = limit.min(readout.output_dim);
    THREAD_READOUT_SESSIONS.with(|cell| {
        let mut sessions = cell.borrow_mut();
        let (index, created) =
            resident_readout_session_index(&mut sessions, readout, features.rows, k)?;
        if !created {
            LINEAR_READOUT_TOPK_SESSION_REUSES.fetch_add(1, Ordering::Relaxed);
        }
        LINEAR_READOUT_TOPK_CALLS.fetch_add(1, Ordering::Relaxed);
        LINEAR_READOUT_TOPK_ROWS.fetch_add(features.rows, Ordering::Relaxed);
        LINEAR_READOUT_TOPK_SESSION_MAX_ENTRIES.fetch_max(sessions.len(), Ordering::Relaxed);
        let session = sessions
            .get_mut(index)
            .ok_or_else(|| SymError::InvalidArgument("CUDA readout session missing".to_string()))?;
        linear_readout_topk_log_probs_cuda_session(session, features, readout, k, top_p)
    })
}

#[cfg(feature = "cuda")]
pub fn linear_readout_topk_log_probs_cuda_session(
    session: &mut CudaLinearReadoutSession,
    features: &Tensor,
    readout: &LinearReadout,
    limit: usize,
    top_p: f32,
) -> Result<Vec<Vec<(usize, f32)>>> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    features.ensure_cols(
        readout.input_dim,
        "CUDA linear readout top-k session features",
    )?;
    if features.rows == 0 || readout.output_dim == 0 || limit == 0 {
        return Ok(vec![Vec::new(); features.rows]);
    }
    let k = limit.min(readout.output_dim);
    if session.ensure_capacity(readout, features.rows, k)? {
        LINEAR_READOUT_TOPK_SESSION_RESIZES.fetch_add(1, Ordering::Relaxed);
    }
    let required_features = features.data.len();
    let required_logits = features.rows.saturating_mul(readout.output_dim);
    let required_topk = features.rows.saturating_mul(k);
    if session.features_dev.len() < required_features
        || session.logits_dev.len() < required_logits
        || session.indices_dev.len() < required_topk
        || session.log_probs_dev.len() < required_topk
    {
        session.row_capacity = 0;
        session.k_capacity = 0;
        if session.ensure_capacity(readout, features.rows, k)? {
            LINEAR_READOUT_TOPK_SESSION_RESIZES.fetch_add(1, Ordering::Relaxed);
        }
    }
    if session.features_dev.len() < required_features
        || session.logits_dev.len() < required_logits
        || session.indices_dev.len() < required_topk
        || session.log_probs_dev.len() < required_topk
    {
        return Err(SymError::Shape(format!(
            "CUDA top-k readout session capacity shortfall: features_dev={} required_features={} logits_dev={} required_logits={} indices_dev={} required_topk={} log_probs_dev={}",
            session.features_dev.len(),
            required_features,
            session.logits_dev.len(),
            required_logits,
            session.indices_dev.len(),
            required_topk,
            session.log_probs_dev.len()
        )));
    }
    session
        .stream
        .memcpy_htod(&features.data, &mut session.features_dev)
        .map_err(cuda_error)?;

    let kernel = readout_kernel_cache()?;
    let stream = &session.stream;
    let sample_count = features.rows as i32;
    let input_dim = readout.input_dim as i32;
    let output_dim = readout.output_dim as i32;
    let top_p = top_p.clamp(0.0, 1.0);

    let logits_cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (features.rows as u32, readout.output_dim as u32, 1),
        shared_mem_bytes: 0,
    };
    let mut logits_args = stream.launch_builder(&kernel.logits_function);
    logits_args.arg(&session.features_dev);
    logits_args.arg(&session.weights_dev);
    logits_args.arg(&session.bias_dev);
    logits_args.arg(&mut session.logits_dev);
    logits_args.arg(&sample_count);
    logits_args.arg(&input_dim);
    logits_args.arg(&output_dim);
    unsafe {
        logits_args.launch(logits_cfg).map_err(cuda_error)?;
    }

    let topk_cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (features.rows as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let k_i32 = k as i32;
    let mut topk_args = stream.launch_builder(&kernel.topk_function);
    topk_args.arg(&session.logits_dev);
    topk_args.arg(&mut session.indices_dev);
    topk_args.arg(&mut session.log_probs_dev);
    topk_args.arg(&sample_count);
    topk_args.arg(&output_dim);
    topk_args.arg(&k_i32);
    topk_args.arg(&top_p);
    unsafe {
        topk_args.launch(topk_cfg).map_err(cuda_error)?;
    }

    let indices = stream
        .clone_dtoh(&session.indices_dev)
        .map_err(cuda_error)?;
    let log_probs = stream
        .clone_dtoh(&session.log_probs_dev)
        .map_err(cuda_error)?;
    let mut rows = Vec::with_capacity(features.rows);
    for sample in 0..features.rows {
        let mut row = Vec::with_capacity(k);
        for rank in 0..k {
            let offset = sample * k + rank;
            let index = indices[offset];
            if index >= 0 {
                row.push((index as usize, log_probs[offset]));
            }
        }
        rows.push(row);
    }
    Ok(rows)
}

#[cfg(feature = "cuda")]
pub fn train_standalone_symliquid_cuda(
    config: symliquid_core::benchmarks::StandaloneTrainConfig,
    samples_per_launch: usize,
) -> Result<symliquid_core::benchmarks::StandaloneTrainReport> {
    use std::collections::BTreeMap;
    use std::time::Instant;

    use rand::SeedableRng;
    use symliquid_core::benchmarks::{
        evaluate_standalone_model, generate_cgs_hard_suite, prepare_training_examples,
        standalone_features, standalone_output_vocab, training_batch_tensor,
        write_readout_artifact, StandaloneTrainReport,
    };

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
    let cuda_readout_start = Instant::now();
    let trace = train_readout_sgd_cuda(
        &features,
        &targets,
        &mut readout,
        config.epochs,
        config.lr,
        samples_per_launch,
    )?;
    timing_breakdown_ms.insert(
        "cuda_readout_train".to_string(),
        cuda_readout_start.elapsed().as_millis(),
    );
    let train_runtime_ms = train_start.elapsed().as_millis();
    let trained_examples = train_examples.len() * config.epochs;
    let train_examples_per_second =
        trained_examples as f32 / (train_runtime_ms.max(1) as f32 / 1000.0);

    let eval_start = Instant::now();
    let eval = evaluate_standalone_model(
        &eval_suite,
        "symliquid-standalone-cuda-readout",
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
            "structured_cgs_vsa_cuda_readout",
        )?;
    }

    Ok(StandaloneTrainReport {
        model_id: "symliquid-standalone-cuda-readout".to_string(),
        feature_set: "structured_cgs_vsa_cuda_readout".to_string(),
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
        runtime_profile: crate::device::runtime_profile(true),
        timing_breakdown_ms,
        kernel_launches: config.epochs * features.rows.div_ceil(samples_per_launch.max(1)),
        cuda_fallback: false,
        eval,
    })
}

#[cfg(feature = "cuda")]
pub fn train_token_superposition_cuda(
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
    let baseline_trace = train_readout_sgd_cuda(
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
    let baseline_examples_seen = config.train_samples * config.baseline_epochs;
    let baseline_kernel_launches =
        config.baseline_epochs * baseline_features.rows.div_ceil(samples_per_launch.max(1));
    let baseline = TokenSuperpositionRunReport {
        id: "baseline_ar_cuda".to_string(),
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

    let mut variants = Vec::new();
    for bag_size in config.bag_sizes.iter().copied().filter(|size| *size > 0) {
        for recovery_ratio in config
            .recovery_ratios
            .iter()
            .copied()
            .filter(|ratio| ratio.is_finite() && *ratio > 0.0 && *ratio < 1.0)
        {
            let variant = train_token_superposition_variant_cuda(
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
            variants.push(variant);
        }
    }

    timing_breakdown_ms.insert("total".to_string(), total_start.elapsed().as_millis());
    Ok(build_token_superposition_report(
        "rust_cuda",
        false,
        config,
        dataset.summary,
        baseline,
        variants,
        timing_breakdown_ms,
    ))
}

#[cfg(feature = "cuda")]
fn train_token_superposition_variant_cuda(
    config: &TokenSuperpositionConfig,
    dataset: &symliquid_core::token_superposition::TokenSuperpositionDataset,
    feature_table: &[Vec<f32>],
    output_dim: usize,
    samples_per_launch: usize,
    baseline: &TokenSuperpositionRunReport,
    recovery_features: &Tensor,
    recovery_targets: &[usize],
    bag_size: usize,
    recovery_ratio: f32,
) -> Result<TokenSuperpositionRunReport> {
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
    let bag_trace = train_readout_bag_sgd_cuda(
        &bag_batch.features,
        &bag_batch.target_bags,
        bag_batch.targets_per_sample,
        &mut readout,
        bag_epochs,
        config.lr,
        samples_per_launch,
    )?;
    let recovery_trace = train_readout_sgd_cuda(
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
    let train_examples_seen = bag_samples * bag_epochs + config.train_samples * recovery_epochs;
    let baseline_examples_seen = baseline.train_examples_seen.max(1);
    let kernel_launches = bag_epochs * bag_batch.features.rows.div_ceil(samples_per_launch.max(1))
        + recovery_epochs * recovery_features.rows.div_ceil(samples_per_launch.max(1));
    Ok(TokenSuperpositionRunReport {
        id: format!("tst_s{bag_size}_r{recovery_ratio:.2}_cuda"),
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
    })
}

#[cfg(feature = "cuda")]
fn examples_per_second(examples: usize, runtime_ms: u128) -> f32 {
    examples as f32 / (runtime_ms.max(1) as f32 / 1000.0)
}

#[cfg(not(feature = "cuda"))]
pub fn train_readout_sgd_cuda(
    _features: &Tensor,
    _targets: &[usize],
    _readout: &mut LinearReadout,
    _epochs: usize,
    _lr: f32,
    _samples_per_launch: usize,
) -> Result<TrainingTrace> {
    Err(SymError::InvalidArgument(
        "CUDA feature is not enabled for readout trainer".to_string(),
    ))
}

#[cfg(not(feature = "cuda"))]
pub fn train_readout_bag_sgd_cuda(
    _features: &Tensor,
    _target_bags: &[usize],
    _targets_per_sample: usize,
    _readout: &mut LinearReadout,
    _epochs: usize,
    _lr: f32,
    _samples_per_launch: usize,
) -> Result<TrainingTrace> {
    Err(SymError::InvalidArgument(
        "CUDA feature is not enabled for bag readout trainer".to_string(),
    ))
}

#[cfg(not(feature = "cuda"))]
pub fn train_code_fast_readout_cuda(
    _features: &Tensor,
    _targets: &[usize],
    _salts: &[usize],
    _readout: &mut LinearReadout,
    _epochs: usize,
    _lr: f32,
) -> Result<TrainingTrace> {
    Err(SymError::InvalidArgument(
        "CUDA feature is not enabled for Code LM fast readout trainer".to_string(),
    ))
}

#[cfg(not(feature = "cuda"))]
pub fn train_standalone_symliquid_cuda(
    _config: symliquid_core::benchmarks::StandaloneTrainConfig,
    _samples_per_launch: usize,
) -> Result<symliquid_core::benchmarks::StandaloneTrainReport> {
    Err(SymError::InvalidArgument(
        "CUDA feature is not enabled for standalone trainer".to_string(),
    ))
}

#[cfg(not(feature = "cuda"))]
pub fn train_token_superposition_cuda(
    _root: &std::path::Path,
    _config: TokenSuperpositionConfig,
    _samples_per_launch: usize,
) -> Result<symliquid_core::token_superposition::TokenSuperpositionReport> {
    Err(SymError::InvalidArgument(
        "CUDA feature is not enabled for token superposition trainer".to_string(),
    ))
}

#[cfg(feature = "cuda")]
fn cuda_error(error: impl std::fmt::Display) -> SymError {
    SymError::InvalidArgument(format!("CUDA readout operation failed: {error}"))
}
