mod rollout;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::ffi::CStr;
use std::fs;
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

#[derive(Debug, Deserialize)]
pub(crate) struct PolicyArtifact {
    #[allow(dead_code)]
    pub(crate) labels: Vec<String>,
    pub(crate) hv_dim: usize,
    pub(crate) output_dim: usize,
    pub(crate) weights: Vec<f32>,
    pub(crate) bias: Vec<f32>,
    #[serde(default = "default_feature_set")]
    pub(crate) feature_set: String,
}

pub struct FfiPolicy {
    pub(crate) artifact: PolicyArtifact,
    pub(crate) memory_state: Vec<f32>,
}

fn memory_stride_for_feature_set(feature_set: &str) -> usize {
    match feature_set {
        "slot_tmaze_recurrent_linear_v1" => 2,
        _ => 1,
    }
}

fn default_feature_set() -> String {
    "numeric_hash_vsa".to_string()
}

#[no_mangle]
/// Load a JSON policy artifact and return an opaque policy handle.
///
/// # Safety
///
/// `path` must be a valid null-terminated string pointer. The returned handle
/// must be released exactly once with `symliquid_policy_free`.
pub unsafe extern "C" fn symliquid_policy_load(
    path: *const c_char,
    num_envs: usize,
) -> *mut FfiPolicy {
    catch_unwind(AssertUnwindSafe(|| {
        if path.is_null() {
            return ptr::null_mut();
        }
        let path = unsafe { CStr::from_ptr(path) }
            .to_string_lossy()
            .into_owned();
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(_) => return ptr::null_mut(),
        };
        let artifact: PolicyArtifact = match serde_json::from_str(&text) {
            Ok(artifact) => artifact,
            Err(_) => return ptr::null_mut(),
        };
        if artifact.hv_dim == 0
            || artifact.output_dim == 0
            || artifact.weights.len() != artifact.hv_dim * artifact.output_dim
            || artifact.bias.len() != artifact.output_dim
        {
            return ptr::null_mut();
        }
        Box::into_raw(Box::new(FfiPolicy {
            memory_state: vec![
                0.0;
                num_envs * memory_stride_for_feature_set(&artifact.feature_set)
            ],
            artifact,
        }))
    }))
    .unwrap_or(ptr::null_mut())
}

#[no_mangle]
/// Construct a policy handle from caller-owned weight and bias arrays.
///
/// # Safety
///
/// `feature_set` must be a valid null-terminated string pointer. `weights`
/// must contain `weights_len == hv_dim * output_dim` valid `f32` values and
/// `bias` must contain `bias_len == output_dim` valid `f32` values. The
/// returned handle must be released exactly once with `symliquid_policy_free`.
pub unsafe extern "C" fn symliquid_policy_from_parts(
    feature_set: *const c_char,
    hv_dim: usize,
    output_dim: usize,
    weights: *const f32,
    weights_len: usize,
    bias: *const f32,
    bias_len: usize,
    num_envs: usize,
) -> *mut FfiPolicy {
    catch_unwind(AssertUnwindSafe(|| {
        if feature_set.is_null()
            || weights.is_null()
            || bias.is_null()
            || hv_dim == 0
            || output_dim == 0
            || weights_len != hv_dim * output_dim
            || bias_len != output_dim
        {
            return ptr::null_mut();
        }
        let feature_set = unsafe { CStr::from_ptr(feature_set) }
            .to_string_lossy()
            .into_owned();
        let memory_stride = memory_stride_for_feature_set(&feature_set);
        let weights = unsafe { std::slice::from_raw_parts(weights, weights_len) }.to_vec();
        let bias = unsafe { std::slice::from_raw_parts(bias, bias_len) }.to_vec();
        let labels = (0..output_dim)
            .map(|idx| format!("action_{idx}"))
            .collect::<Vec<_>>();
        Box::into_raw(Box::new(FfiPolicy {
            artifact: PolicyArtifact {
                labels,
                hv_dim,
                output_dim,
                weights,
                bias,
                feature_set,
            },
            memory_state: vec![0.0; num_envs * memory_stride],
        }))
    }))
    .unwrap_or(ptr::null_mut())
}

#[no_mangle]
/// Release a policy handle allocated by this library.
///
/// # Safety
///
/// `policy` must be null or a handle returned by this library that has not
/// already been freed.
pub unsafe extern "C" fn symliquid_policy_free(policy: *mut FfiPolicy) {
    if !policy.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(policy));
        }));
    }
}

#[no_mangle]
/// Reset the recurrent memory state on a policy handle.
///
/// # Safety
///
/// `policy` must be a valid handle returned by this library.
pub unsafe extern "C" fn symliquid_policy_reset(policy: *mut FfiPolicy, num_envs: usize) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        if policy.is_null() {
            return -1;
        }
        let policy = unsafe { &mut *policy };
        policy.memory_state.clear();
        policy.memory_state.resize(
            num_envs * memory_stride_for_feature_set(&policy.artifact.feature_set),
            0.0,
        );
        0
    }))
    .unwrap_or(-2)
}

#[no_mangle]
/// Return the output dimension for a policy handle, or zero for null handles.
///
/// # Safety
///
/// `policy` must be null or a valid handle returned by this library.
pub unsafe extern "C" fn symliquid_policy_output_dim(policy: *const FfiPolicy) -> usize {
    catch_unwind(AssertUnwindSafe(|| {
        if policy.is_null() {
            return 0;
        }
        unsafe { (*policy).artifact.output_dim }
    }))
    .unwrap_or(0)
}

#[no_mangle]
/// Score a batch of observations and write integer actions to `actions_out`.
///
/// # Safety
///
/// `policy` must be a valid handle returned by this library. `observations`
/// must point to at least `num_envs * obs_dim` valid `f32` values, and
/// `actions_out` must point to at least `num_envs` valid `c_int` slots.
pub unsafe extern "C" fn symliquid_policy_act(
    policy: *mut FfiPolicy,
    observations: *const f32,
    num_envs: usize,
    obs_dim: usize,
    actions_out: *mut c_int,
) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        if policy.is_null()
            || observations.is_null()
            || actions_out.is_null()
            || obs_dim == 0
            || num_envs == 0
        {
            return -1;
        }
        let policy = unsafe { &mut *policy };
        let required_memory =
            num_envs * memory_stride_for_feature_set(&policy.artifact.feature_set);
        if policy.memory_state.len() < required_memory {
            policy.memory_state.resize(required_memory, 0.0);
        }
        let observations = unsafe { std::slice::from_raw_parts(observations, num_envs * obs_dim) };
        let actions_out = unsafe { std::slice::from_raw_parts_mut(actions_out, num_envs) };

        for (env_idx, action_slot) in actions_out.iter_mut().enumerate().take(num_envs) {
            let start = env_idx * obs_dim;
            let end = start + obs_dim;
            let obs = &observations[start..end];
            let action = policy.score_one(obs, env_idx);
            *action_slot = action as c_int;
        }
        0
    }))
    .unwrap_or(-2)
}

#[no_mangle]
/// Train a discrete local Ocean-style policy with the Rust-owned rollout loop.
///
/// # Safety
///
/// `env_name` and `policy_out` must be valid null-terminated string pointers.
/// `report_out` may be null; when non-null, it must also be a valid
/// null-terminated string pointer.
pub unsafe extern "C" fn symliquid_train_discrete_cem(
    env_name: *const c_char,
    policy_out: *const c_char,
    report_out: *const c_char,
    iterations: usize,
    population: usize,
    elite_count: usize,
    num_envs: usize,
    train_steps: usize,
    eval_steps: usize,
    seed: u64,
) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        if env_name.is_null() || policy_out.is_null() {
            return -1;
        }
        let env_name = unsafe { CStr::from_ptr(env_name) }
            .to_string_lossy()
            .into_owned();
        let policy_out = unsafe { CStr::from_ptr(policy_out) }
            .to_string_lossy()
            .into_owned();
        let report_out = if report_out.is_null() {
            None
        } else {
            Some(
                unsafe { CStr::from_ptr(report_out) }
                    .to_string_lossy()
                    .into_owned(),
            )
        };
        match rollout::train_discrete_cem(rollout::TrainConfig {
            env_name,
            policy_out,
            report_out,
            iterations,
            population,
            elite_count,
            num_envs,
            train_steps,
            eval_steps,
            seed,
        }) {
            Ok(()) => 0,
            Err(_) => -2,
        }
    }))
    .unwrap_or(-3)
}

impl FfiPolicy {
    pub(crate) fn score_one(&mut self, observation: &[f32], env_idx: usize) -> usize {
        let features = self.features(observation, env_idx);
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for out_idx in 0..self.artifact.output_dim {
            let mut score = self.artifact.bias[out_idx];
            let row_offset = out_idx * self.artifact.hv_dim;
            for &(idx, value) in &features {
                score += self.artifact.weights[row_offset + idx] * value;
            }
            if score > best_score {
                best_score = score;
                best_idx = out_idx;
            }
        }
        best_idx
    }

    fn features(&mut self, observation: &[f32], env_idx: usize) -> Vec<(usize, f32)> {
        match self.artifact.feature_set.as_str() {
            "cartpole_linear_v1" => cartpole_linear_features(observation, self.artifact.hv_dim),
            "dense_linear_v1" => dense_linear_features(observation, self.artifact.hv_dim),
            "memory_recurrent_linear_v1" => recurrent_linear_features(
                observation,
                self.artifact.hv_dim,
                &mut self.memory_state,
                env_idx,
                RecurrentMode::Memory,
            ),
            "evidence_recurrent_linear_v1" => recurrent_linear_features(
                observation,
                self.artifact.hv_dim,
                &mut self.memory_state,
                env_idx,
                RecurrentMode::Evidence,
            ),
            "evidence_sum_recurrent_linear_v1" => recurrent_linear_features(
                observation,
                self.artifact.hv_dim,
                &mut self.memory_state,
                env_idx,
                RecurrentMode::EvidenceSum,
            ),
            "evidence_tmaze_recurrent_linear_v1" => recurrent_linear_features(
                observation,
                self.artifact.hv_dim,
                &mut self.memory_state,
                env_idx,
                RecurrentMode::EvidenceTMaze,
            ),
            "evidence_sum_tmaze_recurrent_linear_v1" => recurrent_linear_features(
                observation,
                self.artifact.hv_dim,
                &mut self.memory_state,
                env_idx,
                RecurrentMode::EvidenceSumTMaze,
            ),
            "slot_tmaze_recurrent_linear_v1" => slot_tmaze_features(
                observation,
                self.artifact.hv_dim,
                &mut self.memory_state,
                env_idx,
            ),
            "tmaze_recurrent_linear_v1" => recurrent_linear_features(
                observation,
                self.artifact.hv_dim,
                &mut self.memory_state,
                env_idx,
                RecurrentMode::TMaze,
            ),
            _ => numeric_hash_features(observation, self.artifact.hv_dim),
        }
    }
}

fn cartpole_linear_features(observation: &[f32], hv_dim: usize) -> Vec<(usize, f32)> {
    let x = observation.first().copied().unwrap_or(0.0);
    let x_dot = observation.get(1).copied().unwrap_or(0.0);
    let theta = observation.get(2).copied().unwrap_or(0.0);
    let theta_dot = observation.get(3).copied().unwrap_or(0.0);
    dense_to_sparse(
        &[
            1.0,
            x,
            x_dot,
            theta,
            theta_dot,
            x * x,
            theta * theta,
            x_dot * theta_dot,
            theta + 0.35 * theta_dot + 0.03 * x + 0.01 * x_dot,
        ],
        hv_dim,
    )
}

fn dense_linear_features(observation: &[f32], hv_dim: usize) -> Vec<(usize, f32)> {
    let mut dense = Vec::with_capacity(1 + observation.len() * 2 + 1);
    dense.push(1.0);
    dense.extend_from_slice(observation);
    dense.extend(observation.iter().map(|value| value * value));
    if observation.len() >= 2 {
        dense.push(observation[0] * observation[1]);
    }
    dense_to_sparse(&dense, hv_dim)
}

#[derive(Clone, Copy)]
enum RecurrentMode {
    Memory,
    Evidence,
    EvidenceSum,
    EvidenceTMaze,
    EvidenceSumTMaze,
    TMaze,
}

fn recurrent_linear_features(
    observation: &[f32],
    hv_dim: usize,
    memory_state: &mut [f32],
    env_idx: usize,
    mode: RecurrentMode,
) -> Vec<(usize, f32)> {
    let first = observation.first().copied().unwrap_or(0.0);
    match mode {
        RecurrentMode::Memory => {
            if first.abs() > 0.5 {
                memory_state[env_idx] = if first > 0.0 { 1.0 } else { -1.0 };
            }
        }
        RecurrentMode::Evidence
        | RecurrentMode::EvidenceTMaze
        | RecurrentMode::EvidenceSum
        | RecurrentMode::EvidenceSumTMaze => {
            if observation.len() >= 5 && observation[4] > 0.5 {
                memory_state[env_idx] = 0.0;
            }
            if observation.len() >= 2 && observation[1] > 0.5 {
                if matches!(
                    mode,
                    RecurrentMode::EvidenceSum | RecurrentMode::EvidenceSumTMaze
                ) {
                    memory_state[env_idx] += first;
                } else {
                    memory_state[env_idx] = 0.75 * memory_state[env_idx] + first;
                }
            }
        }
        RecurrentMode::TMaze => {
            if (1.75..2.5).contains(&first) {
                memory_state[env_idx] = -1.0;
            } else if first >= 2.5 {
                memory_state[env_idx] = 1.0;
            }
        }
    }
    let memory_value = memory_state[env_idx];
    let mut dense = Vec::with_capacity(2 + observation.len() * 2 + 2);
    dense.push(1.0);
    dense.extend_from_slice(observation);
    dense.push(memory_value);
    dense.extend(observation.iter().map(|value| value * memory_value));
    if matches!(mode, RecurrentMode::TMaze) {
        let at_branch = if observation.len() >= 4
            && observation[1] == 0.0
            && observation[2] > 0.5
            && observation[3] > 0.5
        {
            1.0
        } else {
            0.0
        };
        dense.push(at_branch);
        dense.push(at_branch * memory_value);
    } else if matches!(
        mode,
        RecurrentMode::Evidence
            | RecurrentMode::EvidenceTMaze
            | RecurrentMode::EvidenceSum
            | RecurrentMode::EvidenceSumTMaze
    ) {
        let branch_or_decision_phase = observation.get(2).copied().unwrap_or(0.0);
        dense.push(memory_value * branch_or_decision_phase);
        dense.push(first * memory_value);
        dense.push(branch_or_decision_phase);
    }
    dense_to_sparse(&dense, hv_dim)
}

fn slot_tmaze_features(
    observation: &[f32],
    hv_dim: usize,
    memory_state: &mut [f32],
    env_idx: usize,
) -> Vec<(usize, f32)> {
    let cue = observation.first().copied().unwrap_or(0.0);
    let write_a = observation.get(1).copied().unwrap_or(0.0);
    let write_b = observation.get(2).copied().unwrap_or(0.0);
    let branch = observation.get(3).copied().unwrap_or(0.0);
    let query_a = observation.get(4).copied().unwrap_or(0.0);
    let query_b = observation.get(5).copied().unwrap_or(0.0);
    let time_fraction = observation.get(6).copied().unwrap_or(0.0);
    let reset_phase = observation.get(7).copied().unwrap_or(0.0);
    let offset = env_idx * 2;
    if offset + 1 < memory_state.len() {
        if reset_phase > 0.5 {
            memory_state[offset] = 0.0;
            memory_state[offset + 1] = 0.0;
        }
        if write_a > 0.5 {
            memory_state[offset] = cue;
        }
        if write_b > 0.5 {
            memory_state[offset + 1] = cue;
        }
    }
    let slot_a = memory_state.get(offset).copied().unwrap_or(0.0);
    let slot_b = memory_state.get(offset + 1).copied().unwrap_or(0.0);
    let selected = query_a * slot_a + query_b * slot_b;
    dense_to_sparse(
        &[
            1.0,
            cue,
            write_a,
            write_b,
            branch,
            query_a,
            query_b,
            time_fraction,
            reset_phase,
            slot_a,
            slot_b,
            selected,
            branch * selected,
            branch * slot_a,
            branch * slot_b,
            write_a * cue,
            write_b * cue,
            branch * query_a,
            branch * query_b,
            selected * time_fraction,
            slot_a * query_a,
            slot_b * query_b,
        ],
        hv_dim,
    )
}

fn dense_to_sparse(dense: &[f32], hv_dim: usize) -> Vec<(usize, f32)> {
    dense
        .iter()
        .take(hv_dim)
        .enumerate()
        .map(|(idx, value)| (idx, *value))
        .collect()
}

fn numeric_hash_features(observation: &[f32], hv_dim: usize) -> Vec<(usize, f32)> {
    let mut values = Vec::with_capacity(1 + observation.len() * 2);
    add_sparse_feature(&mut values, hv_dim, "bias", 1.0);
    for (idx, value) in observation.iter().enumerate() {
        let bucketed = (value * 8.0).round().clamp(-16.0, 16.0) as i32;
        add_sparse_feature(
            &mut values,
            hv_dim,
            &format!("obs:{idx}:bucket:{bucketed}"),
            1.0,
        );
        let sign = if *value >= 0.0 { 1 } else { -1 };
        add_sparse_feature(&mut values, hv_dim, &format!("obs:{idx}:sign:{sign}"), 0.25);
    }
    let norm = values
        .iter()
        .map(|(_, value)| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1.0);
    values
        .into_iter()
        .map(|(idx, value)| (idx, value / norm))
        .collect()
}

fn add_sparse_feature(features: &mut Vec<(usize, f32)>, hv_dim: usize, key: &str, value: f32) {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    let idx = (u64::from_le_bytes(bytes) as usize) % hv_dim;
    let sign = if digest[8] & 1 == 0 { 1.0 } else { -1.0 };
    features.push((idx, sign * value));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_features_update_state() {
        let mut memory = vec![0.0];
        let features = recurrent_linear_features(
            &[1.0, 0.0, 0.0, 0.0],
            8,
            &mut memory,
            0,
            RecurrentMode::Memory,
        );
        assert_eq!(memory[0], 1.0);
        assert_eq!(features[5], (5, 1.0));
    }

    #[test]
    fn tmaze_features_mark_branch() {
        let mut memory = vec![1.0];
        let features = recurrent_linear_features(
            &[3.0, 0.0, 1.0, 1.0],
            12,
            &mut memory,
            0,
            RecurrentMode::TMaze,
        );
        assert_eq!(memory[0], 1.0);
        assert_eq!(features[10], (10, 1.0));
        assert_eq!(features[11], (11, 1.0));
    }

    #[test]
    fn slot_tmaze_features_bind_role_to_query() {
        let mut memory = vec![0.0; 2];
        let _ = slot_tmaze_features(
            &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            22,
            &mut memory,
            0,
        );
        let _ = slot_tmaze_features(
            &[-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.0],
            22,
            &mut memory,
            0,
        );
        let query_a = slot_tmaze_features(
            &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            22,
            &mut memory,
            0,
        );
        let query_b = slot_tmaze_features(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            22,
            &mut memory,
            0,
        );
        assert_eq!(memory, vec![1.0, -1.0]);
        assert_eq!(query_a[11], (11, 1.0));
        assert_eq!(query_b[11], (11, -1.0));
        assert_eq!(query_a[12], (12, 1.0));
        assert_eq!(query_b[12], (12, -1.0));
    }

    #[test]
    fn evidence_features_accumulate_and_gate_decision() {
        let mut memory = vec![5.0];
        let _ = recurrent_linear_features(
            &[1.0, 1.0, 0.0, 0.0, 1.0],
            15,
            &mut memory,
            0,
            RecurrentMode::Evidence,
        );
        assert_eq!(memory[0], 1.0);
        let features = recurrent_linear_features(
            &[0.0, 0.0, 1.0, 1.0, 0.0],
            15,
            &mut memory,
            0,
            RecurrentMode::Evidence,
        );
        assert_eq!(features[12], (12, 1.0));
    }

    #[test]
    fn evidence_tmaze_features_share_accumulator() {
        let mut memory = vec![0.0];
        let _ = recurrent_linear_features(
            &[1.0, 1.0, 0.0, 0.0, 1.0],
            15,
            &mut memory,
            0,
            RecurrentMode::EvidenceTMaze,
        );
        let features = recurrent_linear_features(
            &[0.0, 0.0, 1.0, 1.0, 0.0],
            15,
            &mut memory,
            0,
            RecurrentMode::EvidenceTMaze,
        );
        assert_eq!(features[12], (12, 1.0));
        assert_eq!(features[14], (14, 1.0));
    }

    #[test]
    fn evidence_sum_features_do_not_decay_cues() {
        let mut memory = vec![0.0];
        let _ = recurrent_linear_features(
            &[1.0, 1.0, 0.0, 0.0, 1.0],
            15,
            &mut memory,
            0,
            RecurrentMode::EvidenceSum,
        );
        let _ = recurrent_linear_features(
            &[1.0, 1.0, 0.0, 0.0, 0.0],
            15,
            &mut memory,
            0,
            RecurrentMode::EvidenceSum,
        );
        assert_eq!(memory[0], 2.0);
    }
}
