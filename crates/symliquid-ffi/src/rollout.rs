use crate::{FfiPolicy, PolicyArtifact};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use serde::Serialize;
use std::cmp::Ordering;
use std::fs;
use std::path::Path;
use std::time::Instant;

pub(crate) struct TrainConfig {
    pub(crate) env_name: String,
    pub(crate) policy_out: String,
    pub(crate) report_out: Option<String>,
    pub(crate) iterations: usize,
    pub(crate) population: usize,
    pub(crate) elite_count: usize,
    pub(crate) num_envs: usize,
    pub(crate) train_steps: usize,
    pub(crate) eval_steps: usize,
    pub(crate) seed: u64,
}

#[derive(Serialize)]
struct HistoryRow {
    iteration: usize,
    best_mean_reward: f32,
    elite_mean_reward: f32,
    search_std_mean: f32,
}

#[derive(Serialize)]
struct TrainingMeta<'a> {
    algorithm: &'a str,
    env: &'a str,
    initialization: &'a str,
    external_inference_calls: usize,
}

#[derive(Serialize)]
struct ArtifactOut<'a> {
    labels: Vec<String>,
    hv_dim: usize,
    output_dim: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
    feature_set: &'a str,
    training: TrainingMeta<'a>,
}

#[derive(Serialize)]
struct ReportOut {
    algorithm: String,
    env: String,
    policy_out: String,
    iterations: usize,
    population: usize,
    elite_count: usize,
    num_envs: usize,
    train_steps: usize,
    eval_steps: usize,
    seed: u64,
    best_train_mean_reward: f32,
    eval_mean_reward: f32,
    feature_set: String,
    policy_backend: String,
    initialization: String,
    external_inference_calls: usize,
    train_runtime_ms: u128,
    eval_runtime_ms: u128,
    train_transitions: usize,
    eval_transitions: usize,
    train_transitions_per_second: f32,
    eval_transitions_per_second: f32,
    ownership: OwnershipReport,
    history: Vec<HistoryRow>,
}

#[derive(Serialize)]
struct OwnershipReport {
    env_stepping: String,
    rewards: String,
    dones: String,
    recurrent_state: String,
    optimizer_state: String,
    policy_scoring: String,
    python_role: String,
    cuda_migration_status: String,
}

struct EvalConfig<'a> {
    env_name: &'a str,
    feature_set: &'a str,
    hv_dim: usize,
    output_dim: usize,
    num_envs: usize,
    steps: usize,
    seeds: &'a [u64],
}

pub(crate) fn train_discrete_cem(config: TrainConfig) -> Result<(), String> {
    let train_started = Instant::now();
    let action_modulo = local_env_action_modulo(&config.env_name)?;
    let (feature_set, hv_dim) = policy_shape_for_env(&config.env_name)?;
    let population = config.population.max(1);
    let elite_count = config.elite_count.max(1).min(population);
    let output_dim = action_modulo;
    let param_dim = output_dim * hv_dim + output_dim;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut mean = initial_discrete_policy_params(&config.env_name, hv_dim, output_dim)?;
    let mut std = vec![0.5_f32; param_dim];
    let mut best_params = mean.clone();
    let mut best_score = f32::NEG_INFINITY;
    let mut history = Vec::with_capacity(config.iterations);
    let train_seeds = (0..3)
        .map(|idx| config.seed + 101 * idx as u64)
        .collect::<Vec<_>>();

    for iteration in 0..config.iterations {
        let mut candidates = Vec::with_capacity(population);
        for _ in 0..population {
            let params = mean
                .iter()
                .zip(&std)
                .map(|(mu, sigma)| {
                    let z: f32 = rng.sample(StandardNormal);
                    mu + sigma * z
                })
                .collect::<Vec<_>>();
            let score = evaluate_discrete_params(
                &params,
                EvalConfig {
                    env_name: &config.env_name,
                    feature_set,
                    hv_dim,
                    output_dim,
                    num_envs: config.num_envs,
                    steps: config.train_steps,
                    seeds: &train_seeds,
                },
            )?;
            candidates.push((score, params));
        }
        candidates.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));
        let elites = &candidates[..elite_count];
        if elites[0].0 > best_score {
            best_score = elites[0].0;
            best_params.clone_from(&elites[0].1);
        }
        for idx in 0..param_dim {
            let mu = elites.iter().map(|(_, params)| params[idx]).sum::<f32>() / elite_count as f32;
            let variance = elites
                .iter()
                .map(|(_, params)| {
                    let delta = params[idx] - mu;
                    delta * delta
                })
                .sum::<f32>()
                / elite_count as f32;
            mean[idx] = mu;
            std[idx] = (variance.sqrt() * 0.9).max(0.03);
        }
        history.push(HistoryRow {
            iteration,
            best_mean_reward: elites[0].0,
            elite_mean_reward: elites.iter().map(|(score, _)| *score).sum::<f32>()
                / elite_count as f32,
            search_std_mean: std.iter().sum::<f32>() / std.len() as f32,
        });
    }
    let train_runtime_ms = train_started.elapsed().as_millis();

    let eval_seeds = (0..5)
        .map(|idx| config.seed + 10_000 + 211 * idx as u64)
        .collect::<Vec<_>>();
    let eval_started = Instant::now();
    let eval_score = evaluate_discrete_params(
        &best_params,
        EvalConfig {
            env_name: &config.env_name,
            feature_set,
            hv_dim,
            output_dim,
            num_envs: config.num_envs,
            steps: config.eval_steps,
            seeds: &eval_seeds,
        },
    )?;
    let eval_runtime_ms = eval_started.elapsed().as_millis();
    let train_transitions =
        config.iterations * population * train_seeds.len() * config.num_envs * config.train_steps;
    let eval_transitions = eval_seeds.len() * config.num_envs * config.eval_steps;
    write_policy_artifact(
        &config.policy_out,
        &best_params,
        hv_dim,
        output_dim,
        feature_set,
        &config.env_name,
    )?;
    if let Some(report_out) = &config.report_out {
        let report = ReportOut {
            algorithm: "rust_ffi_cross_entropy_search".to_string(),
            env: config.env_name.clone(),
            policy_out: config.policy_out.clone(),
            iterations: config.iterations,
            population,
            elite_count,
            num_envs: config.num_envs,
            train_steps: config.train_steps,
            eval_steps: config.eval_steps,
            seed: config.seed,
            best_train_mean_reward: best_score,
            eval_mean_reward: eval_score,
            feature_set: feature_set.to_string(),
            policy_backend: "rust_ffi_rollout_trainer".to_string(),
            initialization: "cgs_governance_prior".to_string(),
            external_inference_calls: 0,
            train_runtime_ms,
            eval_runtime_ms,
            train_transitions,
            eval_transitions,
            train_transitions_per_second: per_second(train_transitions, train_runtime_ms),
            eval_transitions_per_second: per_second(eval_transitions, eval_runtime_ms),
            ownership: OwnershipReport {
                env_stepping: "rust_owned_cpu".to_string(),
                rewards: "rust_owned_cpu".to_string(),
                dones: "rust_owned_cpu".to_string(),
                recurrent_state: "rust_owned_policy_state".to_string(),
                optimizer_state: "rust_owned_cross_entropy_search".to_string(),
                policy_scoring: "rust_owned_batched_policy_scoring".to_string(),
                python_role: "ffi_orchestration_only".to_string(),
                cuda_migration_status:
                    "next: move env stepping/reward/done/policy scoring buffers into CUDA kernels"
                        .to_string(),
            },
            history,
        };
        write_json(report_out, &report)?;
    }
    Ok(())
}

fn per_second(count: usize, runtime_ms: u128) -> f32 {
    if runtime_ms == 0 {
        return count as f32;
    }
    count as f32 / (runtime_ms as f32 / 1000.0)
}

fn evaluate_discrete_params(params: &[f32], config: EvalConfig<'_>) -> Result<f32, String> {
    let weights_len = config.output_dim * config.hv_dim;
    if params.len() != weights_len + config.output_dim {
        return Err("parameter vector length does not match policy shape".to_string());
    }
    let mut scores = Vec::with_capacity(config.seeds.len());
    for &seed in config.seeds {
        let mut env = LocalEnv::new(config.env_name, config.num_envs, seed)?;
        let mut policy = FfiPolicy {
            artifact: PolicyArtifact {
                labels: (0..config.output_dim)
                    .map(|idx| format!("action_{idx}"))
                    .collect(),
                hv_dim: config.hv_dim,
                output_dim: config.output_dim,
                weights: params[..weights_len].to_vec(),
                bias: params[weights_len..].to_vec(),
                feature_set: config.feature_set.to_string(),
            },
            memory_state: vec![
                0.0;
                config.num_envs
                    * crate::memory_stride_for_feature_set(config.feature_set)
            ],
        };
        let action_modulo = local_env_action_modulo(config.env_name)?;
        let mut observations = env.reset();
        let mut total_reward = 0.0_f32;
        let transitions = (config.num_envs * config.steps).max(1) as f32;
        for _ in 0..config.steps {
            let actions = observations
                .iter()
                .enumerate()
                .map(|(env_idx, observation)| {
                    policy.score_one(observation, env_idx) % action_modulo
                })
                .collect::<Vec<_>>();
            let (next_observations, rewards) = env.step(&actions);
            total_reward += rewards.iter().sum::<f32>();
            observations = next_observations;
        }
        scores.push(total_reward / transitions);
    }
    Ok(scores.iter().sum::<f32>() / scores.len().max(1) as f32)
}

fn write_policy_artifact(
    path: &str,
    params: &[f32],
    hv_dim: usize,
    output_dim: usize,
    feature_set: &str,
    env_name: &str,
) -> Result<(), String> {
    let weights_len = output_dim * hv_dim;
    let artifact = ArtifactOut {
        labels: (0..output_dim).map(|idx| format!("action_{idx}")).collect(),
        hv_dim,
        output_dim,
        weights: params[..weights_len].to_vec(),
        bias: params[weights_len..].to_vec(),
        feature_set,
        training: TrainingMeta {
            algorithm: "rust_ffi_cross_entropy_search",
            env: env_name,
            initialization: "cgs_governance_prior",
            external_inference_calls: 0,
        },
    };
    write_json(path, &artifact)
}

fn write_json<T: Serialize>(path: &str, payload: &T) -> Result<(), String> {
    let path = Path::new(path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
    }
    let text = serde_json::to_string_pretty(payload).map_err(|err| err.to_string())?;
    fs::write(path, format!("{text}\n")).map_err(|err| err.to_string())
}

fn local_env_action_modulo(env_name: &str) -> Result<usize, String> {
    match env_name {
        "ocean-slot-tmaze" | "ocean-tmaze" => Ok(3),
        "ocean-chain" | "ocean-memory" | "ocean-noisy-memory" => Ok(2),
        "ocean-noisy-tmaze" => Ok(3),
        _ => Err(format!("unsupported Rust rollout env: {env_name}")),
    }
}

fn policy_shape_for_env(env_name: &str) -> Result<(&'static str, usize), String> {
    match env_name {
        "ocean-memory" => Ok(("memory_recurrent_linear_v1", 5)),
        "ocean-noisy-memory" => Ok(("evidence_sum_recurrent_linear_v1", 15)),
        "ocean-noisy-tmaze" => Ok(("evidence_sum_tmaze_recurrent_linear_v1", 15)),
        "ocean-slot-tmaze" => Ok(("slot_tmaze_recurrent_linear_v1", 22)),
        "ocean-tmaze" => Ok(("tmaze_recurrent_linear_v1", 12)),
        "ocean-chain" => Ok(("dense_linear_v1", 8)),
        _ => Err(format!("unsupported Rust rollout env: {env_name}")),
    }
}

fn initial_discrete_policy_params(
    env_name: &str,
    hv_dim: usize,
    output_dim: usize,
) -> Result<Vec<f32>, String> {
    let mut params = vec![0.0_f32; output_dim * hv_dim + output_dim];
    let bias_offset = output_dim * hv_dim;
    match env_name {
        "ocean-chain" => params[bias_offset + 1] = 2.0,
        "ocean-memory" => {
            let memory_idx = 2;
            params[memory_idx] = -2.0;
            params[hv_dim + memory_idx] = 2.0;
        }
        "ocean-noisy-memory" => {
            let decision_memory_idx = 12;
            params[decision_memory_idx] = -2.0;
            params[hv_dim + decision_memory_idx] = 2.0;
        }
        "ocean-noisy-tmaze" => {
            const FORWARD: usize = 0;
            const RIGHT: usize = 1;
            const LEFT: usize = 2;
            let branch_memory_idx = 12;
            let branch_phase_idx = 14;
            params[bias_offset + FORWARD] = 1.0;
            params[FORWARD * hv_dim + branch_phase_idx] = -2.0;
            params[bias_offset + RIGHT] = -1.0;
            params[RIGHT * hv_dim + branch_phase_idx] = 1.0;
            params[RIGHT * hv_dim + branch_memory_idx] = 2.0;
            params[bias_offset + LEFT] = -1.0;
            params[LEFT * hv_dim + branch_phase_idx] = 1.0;
            params[LEFT * hv_dim + branch_memory_idx] = -2.0;
        }
        "ocean-slot-tmaze" => {
            const FORWARD: usize = 0;
            const RIGHT: usize = 1;
            const LEFT: usize = 2;
            let branch_phase_idx = 4;
            let selected_memory_idx = 12;
            params[bias_offset + FORWARD] = 1.0;
            params[FORWARD * hv_dim + branch_phase_idx] = -2.0;
            params[bias_offset + RIGHT] = -1.0;
            params[RIGHT * hv_dim + branch_phase_idx] = 1.0;
            params[RIGHT * hv_dim + selected_memory_idx] = 2.0;
            params[bias_offset + LEFT] = -1.0;
            params[LEFT * hv_dim + branch_phase_idx] = 1.0;
            params[LEFT * hv_dim + selected_memory_idx] = -2.0;
        }
        "ocean-tmaze" => {
            const FORWARD: usize = 0;
            const RIGHT: usize = 1;
            const LEFT: usize = 2;
            let at_branch_idx = 10;
            let branch_memory_idx = 11;
            params[bias_offset + FORWARD] = 1.0;
            params[FORWARD * hv_dim + at_branch_idx] = -2.0;
            params[bias_offset + RIGHT] = -1.0;
            params[RIGHT * hv_dim + at_branch_idx] = 1.0;
            params[RIGHT * hv_dim + branch_memory_idx] = 2.0;
            params[bias_offset + LEFT] = -1.0;
            params[LEFT * hv_dim + at_branch_idx] = 1.0;
            params[LEFT * hv_dim + branch_memory_idx] = -2.0;
        }
        _ => return Err(format!("unsupported Rust rollout env: {env_name}")),
    }
    Ok(params)
}

enum LocalEnv {
    Chain(ChainEnv),
    Memory(MemoryEnv),
    NoisyMemory(NoisyMemoryEnv),
    NoisyTMaze(NoisyTMazeEnv),
    SlotTMaze(SlotTMazeEnv),
    TMaze(TMazeEnv),
}

impl LocalEnv {
    fn new(env_name: &str, num_envs: usize, seed: u64) -> Result<Self, String> {
        match env_name {
            "ocean-chain" => Ok(Self::Chain(ChainEnv::new(num_envs))),
            "ocean-memory" => Ok(Self::Memory(MemoryEnv::new(num_envs, seed))),
            "ocean-noisy-memory" => Ok(Self::NoisyMemory(NoisyMemoryEnv::new(num_envs, seed))),
            "ocean-noisy-tmaze" => Ok(Self::NoisyTMaze(NoisyTMazeEnv::new(num_envs, seed))),
            "ocean-slot-tmaze" => Ok(Self::SlotTMaze(SlotTMazeEnv::new(num_envs, seed))),
            "ocean-tmaze" => Ok(Self::TMaze(TMazeEnv::new(num_envs, seed))),
            _ => Err(format!("unsupported Rust rollout env: {env_name}")),
        }
    }

    fn reset(&mut self) -> Vec<Vec<f32>> {
        match self {
            Self::Chain(env) => env.reset(),
            Self::Memory(env) => env.reset(),
            Self::NoisyMemory(env) => env.reset(),
            Self::NoisyTMaze(env) => env.reset(),
            Self::SlotTMaze(env) => env.reset(),
            Self::TMaze(env) => env.reset(),
        }
    }

    fn step(&mut self, actions: &[usize]) -> (Vec<Vec<f32>>, Vec<f32>) {
        match self {
            Self::Chain(env) => env.step(actions),
            Self::Memory(env) => env.step(actions),
            Self::NoisyMemory(env) => env.step(actions),
            Self::NoisyTMaze(env) => env.step(actions),
            Self::SlotTMaze(env) => env.step(actions),
            Self::TMaze(env) => env.step(actions),
        }
    }
}

struct ChainEnv {
    size: usize,
    states: Vec<usize>,
    ticks: Vec<usize>,
}

impl ChainEnv {
    fn new(num_envs: usize) -> Self {
        Self {
            size: 16,
            states: vec![1; num_envs],
            ticks: vec![0; num_envs],
        }
    }

    fn reset(&mut self) -> Vec<Vec<f32>> {
        self.states.fill(1);
        self.ticks.fill(0);
        self.observations()
    }

    fn step(&mut self, actions: &[usize]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut rewards = Vec::with_capacity(actions.len());
        for (env_idx, &action) in actions.iter().enumerate() {
            self.ticks[env_idx] += 1;
            if action % 2 == 0 {
                self.states[env_idx] = self.states[env_idx].saturating_sub(1);
            } else {
                self.states[env_idx] = (self.states[env_idx] + 1).min(self.size - 1);
            }
            let reward = if self.states[env_idx] == 0 {
                0.001
            } else if self.states[env_idx] == self.size - 1 {
                1.0
            } else {
                0.0
            };
            if self.ticks[env_idx] == self.size + 9 {
                self.states[env_idx] = 1;
                self.ticks[env_idx] = 0;
            }
            rewards.push(reward);
        }
        (self.observations(), rewards)
    }

    fn observations(&self) -> Vec<Vec<f32>> {
        let scale = (self.size - 1).max(1) as f32;
        self.states
            .iter()
            .map(|state| vec![*state as f32 / scale])
            .collect()
    }
}

struct MemoryEnv {
    length: usize,
    rng: StdRng,
    goals: Vec<i32>,
    ticks: Vec<usize>,
    visible: Vec<bool>,
}

impl MemoryEnv {
    fn new(num_envs: usize, seed: u64) -> Self {
        let mut env = Self {
            length: 8,
            rng: StdRng::seed_from_u64(seed),
            goals: vec![1; num_envs],
            ticks: vec![0; num_envs],
            visible: vec![true; num_envs],
        };
        for env_idx in 0..num_envs {
            env.reset_one(env_idx);
        }
        env
    }

    fn reset(&mut self) -> Vec<Vec<f32>> {
        for env_idx in 0..self.goals.len() {
            self.reset_one(env_idx);
        }
        self.observations()
    }

    fn step(&mut self, actions: &[usize]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut rewards = Vec::with_capacity(actions.len());
        for (env_idx, &action) in actions.iter().enumerate() {
            self.visible[env_idx] = false;
            self.ticks[env_idx] += 1;
            let terminal = self.ticks[env_idx] >= self.length;
            let mut reward = 0.0;
            if terminal {
                let correct = (action % 2 == 0 && self.goals[env_idx] == -1)
                    || (action % 2 == 1 && self.goals[env_idx] == 1);
                reward = if correct { 1.0 } else { 0.0 };
                self.reset_one(env_idx);
            }
            rewards.push(reward);
        }
        (self.observations(), rewards)
    }

    fn reset_one(&mut self, env_idx: usize) {
        self.goals[env_idx] = if self.rng.gen_range(0..2) == 0 { -1 } else { 1 };
        self.ticks[env_idx] = 0;
        self.visible[env_idx] = true;
    }

    fn observations(&self) -> Vec<Vec<f32>> {
        self.goals
            .iter()
            .zip(&self.visible)
            .map(|(goal, visible)| vec![if *visible { *goal as f32 } else { 0.0 }])
            .collect()
    }
}

struct NoisyMemoryEnv {
    length: usize,
    evidence_steps: usize,
    evidence_accuracy: f32,
    rng: StdRng,
    goals: Vec<i32>,
    ticks: Vec<usize>,
    evidence: Vec<Vec<f32>>,
    distractors: Vec<Vec<f32>>,
}

impl NoisyMemoryEnv {
    fn new(num_envs: usize, seed: u64) -> Self {
        let length = 12;
        let evidence_steps = 5;
        let mut env = Self {
            length,
            evidence_steps,
            evidence_accuracy: 0.75,
            rng: StdRng::seed_from_u64(seed),
            goals: vec![1; num_envs],
            ticks: vec![0; num_envs],
            evidence: vec![vec![0.0; evidence_steps]; num_envs],
            distractors: vec![vec![0.0; length]; num_envs],
        };
        for env_idx in 0..num_envs {
            env.reset_one(env_idx);
        }
        env
    }

    fn reset(&mut self) -> Vec<Vec<f32>> {
        for env_idx in 0..self.goals.len() {
            self.reset_one(env_idx);
        }
        self.observations()
    }

    fn step(&mut self, actions: &[usize]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut rewards = Vec::with_capacity(actions.len());
        for (env_idx, &action) in actions.iter().enumerate() {
            let decision_tick = self.ticks[env_idx] >= self.length - 1;
            let mut reward = 0.0;
            if decision_tick {
                let correct_action = if self.goals[env_idx] > 0 { 1 } else { 0 };
                reward = if action % 2 == correct_action {
                    1.0
                } else {
                    0.0
                };
                self.reset_one(env_idx);
            } else {
                self.ticks[env_idx] += 1;
            }
            rewards.push(reward);
        }
        (self.observations(), rewards)
    }

    fn reset_one(&mut self, env_idx: usize) {
        let goal = if self.rng.gen_range(0..2) == 0 { -1 } else { 1 };
        self.goals[env_idx] = goal;
        self.ticks[env_idx] = 0;
        for step in 0..self.evidence_steps {
            let sample = if self.rng.gen::<f32>() < self.evidence_accuracy {
                goal
            } else {
                -goal
            };
            self.evidence[env_idx][step] = sample as f32;
        }
        for step in 0..self.length {
            self.distractors[env_idx][step] = if self.rng.gen_range(0..2) == 0 {
                -1.0
            } else {
                1.0
            };
        }
    }

    fn observations(&self) -> Vec<Vec<f32>> {
        self.ticks
            .iter()
            .enumerate()
            .map(|(env_idx, &tick)| {
                let reset_phase = if tick == 0 { 1.0 } else { 0.0 };
                let time_fraction = tick as f32 / (self.length - 1).max(1) as f32;
                if tick < self.evidence_steps {
                    vec![
                        self.evidence[env_idx][tick],
                        1.0,
                        0.0,
                        time_fraction,
                        reset_phase,
                    ]
                } else if tick >= self.length - 1 {
                    vec![0.0, 0.0, 1.0, time_fraction, reset_phase]
                } else {
                    vec![
                        0.25 * self.distractors[env_idx][tick],
                        0.0,
                        0.0,
                        time_fraction,
                        reset_phase,
                    ]
                }
            })
            .collect()
    }
}

struct NoisyTMazeEnv {
    size: usize,
    evidence_steps: usize,
    evidence_accuracy: f32,
    rng: StdRng,
    states: Vec<usize>,
    goals: Vec<i32>,
    ticks: Vec<usize>,
    evidence: Vec<Vec<f32>>,
    distractors: Vec<Vec<f32>>,
}

impl NoisyTMazeEnv {
    const FORWARD: usize = 0;
    const RIGHT: usize = 1;
    const LEFT: usize = 2;

    fn new(num_envs: usize, seed: u64) -> Self {
        let size = 10;
        let evidence_steps = 5;
        let mut env = Self {
            size,
            evidence_steps,
            evidence_accuracy: 0.70,
            rng: StdRng::seed_from_u64(seed),
            states: vec![0; num_envs],
            goals: vec![1; num_envs],
            ticks: vec![0; num_envs],
            evidence: vec![vec![0.0; evidence_steps]; num_envs],
            distractors: vec![vec![0.0; size]; num_envs],
        };
        for env_idx in 0..num_envs {
            env.reset_one(env_idx);
        }
        env
    }

    fn reset(&mut self) -> Vec<Vec<f32>> {
        for env_idx in 0..self.states.len() {
            self.reset_one(env_idx);
        }
        self.observations()
    }

    fn step(&mut self, actions: &[usize]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut rewards = Vec::with_capacity(actions.len());
        for (env_idx, &raw_action) in actions.iter().enumerate() {
            self.ticks[env_idx] += 1;
            let action = raw_action % 3;
            let mut reward = 0.0;
            if self.states[env_idx] == self.size - 1 {
                if action == Self::LEFT || action == Self::RIGHT {
                    let correct_action = if self.goals[env_idx] > 0 {
                        Self::RIGHT
                    } else {
                        Self::LEFT
                    };
                    reward = if action == correct_action { 1.0 } else { 0.0 };
                    self.reset_one(env_idx);
                } else if self.ticks[env_idx] > self.size + 4 {
                    self.reset_one(env_idx);
                }
            } else if action == Self::FORWARD {
                self.states[env_idx] += 1;
            } else if self.ticks[env_idx] > self.size + 4 {
                self.reset_one(env_idx);
            }
            rewards.push(reward);
        }
        (self.observations(), rewards)
    }

    fn reset_one(&mut self, env_idx: usize) {
        let goal = if self.rng.gen_range(0..2) == 0 { -1 } else { 1 };
        self.states[env_idx] = 0;
        self.goals[env_idx] = goal;
        self.ticks[env_idx] = 0;
        for step in 0..self.evidence_steps {
            let sample = if self.rng.gen::<f32>() < self.evidence_accuracy {
                goal
            } else {
                -goal
            };
            self.evidence[env_idx][step] = sample as f32;
        }
        for step in 0..self.size {
            self.distractors[env_idx][step] = if self.rng.gen_range(0..2) == 0 {
                -1.0
            } else {
                1.0
            };
        }
    }

    fn observations(&self) -> Vec<Vec<f32>> {
        self.states
            .iter()
            .enumerate()
            .map(|(env_idx, &state)| {
                let reset_phase = if state == 0 { 1.0 } else { 0.0 };
                let time_fraction = state as f32 / (self.size - 1).max(1) as f32;
                if state < self.evidence_steps {
                    vec![
                        self.evidence[env_idx][state],
                        1.0,
                        0.0,
                        time_fraction,
                        reset_phase,
                    ]
                } else if state == self.size - 1 {
                    vec![0.0, 0.0, 1.0, time_fraction, reset_phase]
                } else {
                    vec![
                        0.25 * self.distractors[env_idx][state],
                        0.0,
                        0.0,
                        time_fraction,
                        reset_phase,
                    ]
                }
            })
            .collect()
    }
}

struct SlotTMazeEnv {
    size: usize,
    rng: StdRng,
    states: Vec<usize>,
    slot_a: Vec<i32>,
    slot_b: Vec<i32>,
    query_slot: Vec<usize>,
    ticks: Vec<usize>,
    distractors: Vec<Vec<f32>>,
}

impl SlotTMazeEnv {
    const FORWARD: usize = 0;
    const RIGHT: usize = 1;
    const LEFT: usize = 2;

    fn new(num_envs: usize, seed: u64) -> Self {
        let size = 10;
        let mut env = Self {
            size,
            rng: StdRng::seed_from_u64(seed),
            states: vec![0; num_envs],
            slot_a: vec![1; num_envs],
            slot_b: vec![-1; num_envs],
            query_slot: vec![0; num_envs],
            ticks: vec![0; num_envs],
            distractors: vec![vec![0.0; size]; num_envs],
        };
        for env_idx in 0..num_envs {
            env.reset_one(env_idx);
        }
        env
    }

    fn reset(&mut self) -> Vec<Vec<f32>> {
        for env_idx in 0..self.states.len() {
            self.reset_one(env_idx);
        }
        self.observations()
    }

    fn step(&mut self, actions: &[usize]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut rewards = Vec::with_capacity(actions.len());
        for (env_idx, &raw_action) in actions.iter().enumerate() {
            self.ticks[env_idx] += 1;
            let action = raw_action % 3;
            let mut reward = 0.0;
            if self.states[env_idx] == self.size - 1 {
                if action == Self::LEFT || action == Self::RIGHT {
                    let target = if self.query_slot[env_idx] == 0 {
                        self.slot_a[env_idx]
                    } else {
                        self.slot_b[env_idx]
                    };
                    let correct_action = if target > 0 { Self::RIGHT } else { Self::LEFT };
                    reward = if action == correct_action { 1.0 } else { 0.0 };
                    self.reset_one(env_idx);
                } else if self.ticks[env_idx] > self.size + 4 {
                    self.reset_one(env_idx);
                }
            } else if action == Self::FORWARD {
                self.states[env_idx] += 1;
            } else if self.ticks[env_idx] > self.size + 4 {
                self.reset_one(env_idx);
            }
            rewards.push(reward);
        }
        (self.observations(), rewards)
    }

    fn reset_one(&mut self, env_idx: usize) {
        self.states[env_idx] = 0;
        self.ticks[env_idx] = 0;
        self.slot_a[env_idx] = if self.rng.gen_range(0..2) == 0 { -1 } else { 1 };
        self.slot_b[env_idx] = if self.rng.gen_range(0..2) == 0 { -1 } else { 1 };
        self.query_slot[env_idx] = self.rng.gen_range(0..2);
        for step in 0..self.size {
            self.distractors[env_idx][step] = if self.rng.gen_range(0..2) == 0 {
                -1.0
            } else {
                1.0
            };
        }
    }

    fn observations(&self) -> Vec<Vec<f32>> {
        self.states
            .iter()
            .enumerate()
            .map(|(env_idx, &state)| {
                let reset_phase = if state == 0 { 1.0 } else { 0.0 };
                let time_fraction = state as f32 / (self.size - 1).max(1) as f32;
                if state == 0 {
                    vec![
                        self.slot_a[env_idx] as f32,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        time_fraction,
                        reset_phase,
                    ]
                } else if state == 1 {
                    vec![
                        self.slot_b[env_idx] as f32,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        time_fraction,
                        reset_phase,
                    ]
                } else if state == self.size - 1 {
                    let query_a = if self.query_slot[env_idx] == 0 {
                        1.0
                    } else {
                        0.0
                    };
                    let query_b = if self.query_slot[env_idx] == 1 {
                        1.0
                    } else {
                        0.0
                    };
                    vec![
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        query_a,
                        query_b,
                        time_fraction,
                        reset_phase,
                    ]
                } else {
                    vec![
                        0.25 * self.distractors[env_idx][state],
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        time_fraction,
                        reset_phase,
                    ]
                }
            })
            .collect()
    }
}

struct TMazeEnv {
    size: usize,
    rng: StdRng,
    states: Vec<usize>,
    starting_states: Vec<usize>,
    ticks: Vec<usize>,
}

impl TMazeEnv {
    const FORWARD: usize = 0;
    const RIGHT: usize = 1;
    const LEFT: usize = 2;

    fn new(num_envs: usize, seed: u64) -> Self {
        let mut env = Self {
            size: 8,
            rng: StdRng::seed_from_u64(seed),
            states: vec![0; num_envs],
            starting_states: vec![2; num_envs],
            ticks: vec![0; num_envs],
        };
        for env_idx in 0..num_envs {
            env.reset_one(env_idx);
        }
        env
    }

    fn reset(&mut self) -> Vec<Vec<f32>> {
        for env_idx in 0..self.states.len() {
            self.reset_one(env_idx);
        }
        self.observations()
    }

    fn step(&mut self, actions: &[usize]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut rewards = Vec::with_capacity(actions.len());
        for (env_idx, &raw_action) in actions.iter().enumerate() {
            self.ticks[env_idx] += 1;
            let action = raw_action % 3;
            let mut reward = 0.0;
            if self.states[env_idx] == self.size - 1 {
                if action == Self::LEFT || action == Self::RIGHT {
                    let left_reward = if self.starting_states[env_idx] == 2 {
                        1.0
                    } else {
                        -1.0
                    };
                    let right_reward = if self.starting_states[env_idx] == 3 {
                        1.0
                    } else {
                        -1.0
                    };
                    reward = if action == Self::LEFT {
                        left_reward
                    } else {
                        right_reward
                    };
                    self.reset_one(env_idx);
                } else if self.ticks[env_idx] > self.size + 4 {
                    self.reset_one(env_idx);
                }
            } else if action == Self::FORWARD {
                self.states[env_idx] += 1;
            } else if self.ticks[env_idx] > self.size + 4 {
                self.reset_one(env_idx);
            }
            rewards.push(reward);
        }
        (self.observations(), rewards)
    }

    fn reset_one(&mut self, env_idx: usize) {
        self.states[env_idx] = 0;
        self.starting_states[env_idx] = 2 + self.rng.gen_range(0..2);
        self.ticks[env_idx] = 0;
    }

    fn observations(&self) -> Vec<Vec<f32>> {
        self.states
            .iter()
            .zip(&self.starting_states)
            .map(|(state, starting_state)| {
                if *state == 0 {
                    vec![*starting_state as f32, 1.0, 0.0, 0.0]
                } else if *state == self.size - 1 {
                    vec![1.0, 0.0, 1.0, 1.0]
                } else {
                    vec![1.0, 1.0, 0.0, 0.0]
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_rollout_trains_noisy_memory_smoke() {
        let dir = std::env::temp_dir();
        let nonce = std::process::id();
        let policy = dir.join(format!("symliquid_ffi_policy_{nonce}.json"));
        let report = dir.join(format!("symliquid_ffi_report_{nonce}.json"));
        train_discrete_cem(TrainConfig {
            env_name: "ocean-noisy-memory".to_string(),
            policy_out: policy.to_string_lossy().into_owned(),
            report_out: Some(report.to_string_lossy().into_owned()),
            iterations: 1,
            population: 4,
            elite_count: 2,
            num_envs: 8,
            train_steps: 16,
            eval_steps: 16,
            seed: 0,
        })
        .expect("rust noisy-memory trainer should run");
        assert!(policy.exists());
        assert!(report.exists());
        let _ = fs::remove_file(policy);
        let _ = fs::remove_file(report);
    }

    #[test]
    fn chain_policy_prior_scores_positive_reward() {
        let env_name = "ocean-chain";
        let output_dim = local_env_action_modulo(env_name).unwrap();
        let (feature_set, hv_dim) = policy_shape_for_env(env_name).unwrap();
        let params = initial_discrete_policy_params(env_name, hv_dim, output_dim).unwrap();
        let seeds = [0];
        let score = evaluate_discrete_params(
            &params,
            EvalConfig {
                env_name,
                feature_set,
                hv_dim,
                output_dim,
                num_envs: 16,
                steps: 64,
                seeds: &seeds,
            },
        )
        .unwrap();
        assert!(score > 0.0);
    }

    #[test]
    fn noisy_tmaze_prior_scores_unsaturated_reward() {
        let env_name = "ocean-noisy-tmaze";
        let output_dim = local_env_action_modulo(env_name).unwrap();
        let (feature_set, hv_dim) = policy_shape_for_env(env_name).unwrap();
        let params = initial_discrete_policy_params(env_name, hv_dim, output_dim).unwrap();
        let seeds = [0];
        let score = evaluate_discrete_params(
            &params,
            EvalConfig {
                env_name,
                feature_set,
                hv_dim,
                output_dim,
                num_envs: 16,
                steps: 64,
                seeds: &seeds,
            },
        )
        .unwrap();
        assert!(score > 0.0);
        assert!(score < 0.2);
    }

    #[test]
    fn slot_tmaze_prior_solves_delayed_role_query() {
        let env_name = "ocean-slot-tmaze";
        let output_dim = local_env_action_modulo(env_name).unwrap();
        let (feature_set, hv_dim) = policy_shape_for_env(env_name).unwrap();
        let params = initial_discrete_policy_params(env_name, hv_dim, output_dim).unwrap();
        let seeds = [0];
        let score = evaluate_discrete_params(
            &params,
            EvalConfig {
                env_name,
                feature_set,
                hv_dim,
                output_dim,
                num_envs: 32,
                steps: 128,
                seeds: &seeds,
            },
        )
        .unwrap();
        assert!(score > 0.09);
    }
}
