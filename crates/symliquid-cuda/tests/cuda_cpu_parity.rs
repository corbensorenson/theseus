use rand::SeedableRng;
use symliquid_core::modules::vsa::VSAMemory;
#[cfg(feature = "cuda")]
use symliquid_core::{tensor::Tensor, train::LinearReadout};
#[cfg(feature = "cuda")]
use symliquid_cuda::rollout_cuda::{
    rollout_sequence_cpu, rollout_sequence_cuda, RolloutConfig, RolloutParams, RolloutState,
};
use symliquid_cuda::CudaBackend;

#[test]
fn cuda_surface_matches_cpu_vsa_primitives() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(9);
    let a = VSAMemory::random_bipolar(1, 64, &mut rng);
    let b = VSAMemory::random_bipolar(1, 64, &mut rng);
    let backend = CudaBackend::new();

    let cpu_bind = VSAMemory::bind(&a, &b).unwrap();
    let cuda_bind = backend.vsa_bind(&a, &b).unwrap();
    assert_eq!(cpu_bind, cuda_bind);

    let cpu_bundle = VSAMemory::bundle(&a, &b, 0.5).unwrap();
    let cuda_bundle = backend.vsa_bundle(&a, &b, 0.5).unwrap();
    assert_eq!(cpu_bundle, cuda_bundle);

    let cpu_permute = VSAMemory::permute(&a, 7);
    let cuda_permute = backend.vsa_permute(&a, 7);
    assert_eq!(cpu_permute, cuda_permute);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_readout_sgd_matches_cpu_for_micro_steps() {
    let features = Tensor::new(
        4,
        5,
        vec![
            1.0, 0.2, -0.1, 0.4, 0.0, //
            0.0, -0.3, 0.8, 0.1, 0.2, //
            -0.2, 0.7, 0.0, -0.5, 0.3, //
            0.6, 0.0, 0.2, 0.2, -0.4,
        ],
    )
    .unwrap();
    let targets = vec![0usize, 2, 1, 2];
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut cpu = LinearReadout::new(5, 3, &mut rng);
    let mut cuda = cpu.clone();

    for _ in 0..3 {
        for (row, target) in targets.iter().copied().enumerate().take(features.rows) {
            let sample = Tensor::new(1, features.cols, features.row(row).to_vec()).unwrap();
            cpu.train_step(&sample, target, 0.07).unwrap();
        }
    }
    let _trace = symliquid_cuda::readout_cuda::train_readout_sgd_cuda(
        &features, &targets, &mut cuda, 3, 0.07, 2,
    )
    .unwrap();

    for (left, right) in cpu.weights.iter().zip(&cuda.weights) {
        assert!((left - right).abs() < 1e-4, "{left} != {right}");
    }
    for (left, right) in cpu.bias.iter().zip(&cuda.bias) {
        assert!((left - right).abs() < 1e-4, "{left} != {right}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_binary_readout_scoring_matches_cpu_logits_delta() {
    let features = Tensor::new(
        3,
        4,
        vec![
            0.5, -0.2, 0.7, 0.0, //
            -0.1, 0.4, 0.2, -0.6, //
            0.0, 0.3, -0.5, 0.8,
        ],
    )
    .unwrap();
    let mut readout = LinearReadout::zeros(4, 2);
    readout.weights = vec![
        0.1, -0.2, 0.3, 0.0, //
        -0.4, 0.5, 0.2, 0.7,
    ];
    readout.bias = vec![0.05, -0.15];

    let cuda_scores =
        symliquid_cuda::readout_cuda::score_binary_readout_cuda(&features, &readout).unwrap();
    let logits = readout.logits(&features).unwrap();
    for (row, score) in cuda_scores.iter().copied().enumerate() {
        let expected = logits.get(row, 1) - logits.get(row, 0);
        assert!((score - expected).abs() < 1e-5, "{score} != {expected}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_weighted_feature_scores_match_cpu_dot_product() {
    symliquid_cuda::readout_cuda::clear_thread_readout_session();
    let before = symliquid_cuda::readout_cuda::linear_readout_runtime_summary_cuda();
    let features = Tensor::new(
        4,
        5,
        vec![
            1.0, 0.2, -0.1, 0.4, 0.0, //
            0.0, -0.3, 0.8, 0.1, 0.2, //
            -0.2, 0.7, 0.0, -0.5, 0.3, //
            0.6, 0.0, 0.2, 0.2, -0.4,
        ],
    )
    .unwrap();
    let weights = vec![0.7, -0.4, 1.2, 0.1, -0.9];
    let bias = 0.35;

    let cuda =
        symliquid_cuda::readout_cuda::weighted_feature_scores_cuda(&features, &weights, bias)
            .unwrap();
    let cuda_reuse =
        symliquid_cuda::readout_cuda::weighted_feature_scores_cuda(&features, &weights, bias)
            .unwrap();
    assert_eq!(cuda, cuda_reuse);
    for row in 0..features.rows {
        let expected = bias
            + features
                .row(row)
                .iter()
                .zip(&weights)
                .map(|(feature, weight)| feature * weight)
                .sum::<f32>();
        assert!(
            (cuda[row] - expected).abs() < 1e-5,
            "{} != {}",
            cuda[row],
            expected
        );
    }
    let after = symliquid_cuda::readout_cuda::linear_readout_runtime_summary_cuda();
    assert!(
        after.weighted_score_session_create_count > before.weighted_score_session_create_count,
        "weighted ranker session should be created"
    );
    assert!(
        after.weighted_score_session_reuse_count > before.weighted_score_session_reuse_count,
        "weighted ranker session should be reused"
    );
    assert_eq!(after.weighted_score_thread_session_count, 1);
    assert!(
        after.weighted_score_call_count >= before.weighted_score_call_count + 2,
        "weighted ranker calls should be counted"
    );
    assert!(
        after.weighted_score_row_count >= before.weighted_score_row_count + features.rows * 2,
        "weighted ranker scored rows should be counted"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_linear_readout_logits_match_cpu_batch_logits() {
    let features = Tensor::new(
        4,
        3,
        vec![
            1.0, 0.0, -0.5, //
            0.2, 0.8, 0.1, //
            -0.4, 0.3, 0.9, //
            0.0, -0.7, 0.6,
        ],
    )
    .unwrap();
    let mut readout = LinearReadout::zeros(3, 5);
    readout.weights = vec![
        0.1, 0.2, -0.1, //
        -0.3, 0.4, 0.2, //
        0.5, -0.2, 0.0, //
        0.0, 0.1, 0.3, //
        -0.4, -0.1, 0.6,
    ];
    readout.bias = vec![0.0, 0.1, -0.2, 0.3, -0.4];

    let cpu = readout.logits(&features).unwrap();
    let cuda =
        symliquid_cuda::readout_cuda::linear_readout_logits_cuda(&features, &readout).unwrap();
    assert_eq!(cpu.shape(), cuda.shape());
    for (left, right) in cpu.data.iter().zip(&cuda.data) {
        assert!((left - right).abs() < 1e-5, "{left} != {right}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_linear_readout_topk_log_probs_match_cpu_logits() {
    let features = Tensor::new(
        2,
        3,
        vec![
            0.4, -0.6, 0.2, //
            -0.1, 0.9, 0.3,
        ],
    )
    .unwrap();
    let mut readout = LinearReadout::zeros(3, 4);
    readout.weights = vec![
        0.2, -0.1, 0.4, //
        -0.3, 0.5, 0.1, //
        0.6, 0.0, -0.2, //
        -0.4, -0.2, 0.3,
    ];
    readout.bias = vec![0.1, -0.2, 0.05, 0.3];

    let cuda = symliquid_cuda::readout_cuda::linear_readout_topk_log_probs_cuda(
        &features, &readout, 3, 1.0,
    )
    .unwrap();
    let logits = readout.logits(&features).unwrap();
    for (row_idx, cuda_row) in cuda.iter().enumerate() {
        let row = logits.row(row_idx);
        let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = row
            .iter()
            .map(|value| (*value - max_logit).exp())
            .sum::<f32>()
            .max(1e-8);
        let mut expected = row
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, value)| (idx, value - max_logit - denom.ln()))
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap());
        for ((left_idx, left_score), (right_idx, right_score)) in
            cuda_row.iter().copied().zip(expected.into_iter().take(3))
        {
            assert_eq!(left_idx, right_idx);
            assert!(
                (left_score - right_score).abs() < 1e-5,
                "{left_score} != {right_score}"
            );
        }
    }
    let larger_batch = Tensor::new(
        3,
        3,
        vec![
            0.4, -0.6, 0.2, //
            -0.1, 0.9, 0.3, //
            0.7, 0.2, -0.4,
        ],
    )
    .unwrap();
    let larger_cuda = symliquid_cuda::readout_cuda::linear_readout_topk_log_probs_cuda(
        &larger_batch,
        &readout,
        3,
        1.0,
    )
    .unwrap();
    let logits = readout.logits(&larger_batch).unwrap();
    for (row_idx, cuda_row) in larger_cuda.iter().enumerate() {
        let row = logits.row(row_idx);
        let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = row
            .iter()
            .map(|value| (*value - max_logit).exp())
            .sum::<f32>()
            .max(1e-8);
        let mut expected = row
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, value)| (idx, value - max_logit - denom.ln()))
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap());
        for ((left_idx, left_score), (right_idx, right_score)) in
            cuda_row.iter().copied().zip(expected.into_iter().take(3))
        {
            assert_eq!(left_idx, right_idx);
            assert!(
                (left_score - right_score).abs() < 1e-5,
                "{left_score} != {right_score}"
            );
        }
    }
    let summary = symliquid_cuda::readout_cuda::linear_readout_runtime_summary_cuda();
    assert!(
        summary.topk_session_reuse_count >= 1,
        "expected same-readout top-k call to reuse the resident CUDA session: {summary:?}"
    );
    assert!(
        summary.topk_call_count >= 2,
        "expected top-k telemetry to count both calls: {summary:?}"
    );
    assert!(
        summary.topk_row_count >= 5,
        "expected top-k telemetry to count processed rows: {summary:?}"
    );
    symliquid_cuda::readout_cuda::clear_thread_readout_session();
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_linear_readout_topk_keeps_multiple_readouts_resident_per_thread() {
    let cache_limit = std::env::var("THESEUS_CUDA_READOUT_SESSION_CACHE_LIMIT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(4);
    if cache_limit < 2 {
        return;
    }
    symliquid_cuda::readout_cuda::clear_thread_readout_session();
    let features = Tensor::new(
        2,
        3,
        vec![
            0.4, -0.6, 0.2, //
            -0.1, 0.9, 0.3,
        ],
    )
    .unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(314);
    let readout_a = LinearReadout::new(3, 5, &mut rng);
    let readout_b = LinearReadout::new(3, 5, &mut rng);
    let before = symliquid_cuda::readout_cuda::linear_readout_runtime_summary_cuda();

    symliquid_cuda::readout_cuda::prepare_thread_readout_session_cuda(&readout_a, 8, 3).unwrap();
    symliquid_cuda::readout_cuda::linear_readout_topk_log_probs_cuda(&features, &readout_a, 3, 1.0)
        .unwrap();
    symliquid_cuda::readout_cuda::linear_readout_topk_log_probs_cuda(&features, &readout_b, 3, 1.0)
        .unwrap();
    symliquid_cuda::readout_cuda::linear_readout_topk_log_probs_cuda(&features, &readout_a, 3, 1.0)
        .unwrap();

    let after = symliquid_cuda::readout_cuda::linear_readout_runtime_summary_cuda();
    assert!(
        after.topk_thread_session_count >= 2,
        "expected two resident readout sessions in this worker thread: {after:?}"
    );
    assert!(
        after.topk_session_max_entry_count >= 2,
        "expected telemetry to observe a multi-readout resident cache: {after:?}"
    );
    assert!(
        after.topk_session_prepare_count > before.topk_session_prepare_count,
        "expected explicit resident-session preparation to be counted: before={before:?} after={after:?}"
    );
    assert!(
        after.topk_session_reuse_count > before.topk_session_reuse_count,
        "expected alternating readout calls to reuse a resident session: before={before:?} after={after:?}"
    );
    symliquid_cuda::readout_cuda::clear_thread_readout_session();
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_bag_readout_sgd_matches_cpu_for_micro_steps() {
    let features = Tensor::new(
        3,
        4,
        vec![
            0.9, 0.1, -0.2, 0.0, //
            -0.3, 0.5, 0.2, 0.4, //
            0.1, -0.7, 0.3, 0.8,
        ],
    )
    .unwrap();
    let target_bags = vec![0usize, 2, 1, 2, 2, 3];
    let mut rng = rand::rngs::StdRng::seed_from_u64(99);
    let mut cpu = LinearReadout::new(4, 4, &mut rng);
    let mut cuda = cpu.clone();

    for _ in 0..4 {
        for row in 0..features.rows {
            let sample = Tensor::new(1, features.cols, features.row(row).to_vec()).unwrap();
            let bag = &target_bags[row * 2..(row + 1) * 2];
            cpu.train_batch_target_bags(&sample, bag, 2, 0.05).unwrap();
        }
    }
    let _trace = symliquid_cuda::readout_cuda::train_readout_bag_sgd_cuda(
        &features,
        &target_bags,
        2,
        &mut cuda,
        4,
        0.05,
        2,
    )
    .unwrap();

    for (left, right) in cpu.weights.iter().zip(&cuda.weights) {
        assert!((left - right).abs() < 1e-4, "{left} != {right}");
    }
    for (left, right) in cpu.bias.iter().zip(&cuda.bias) {
        assert!((left - right).abs() < 1e-4, "{left} != {right}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rollout_state_update_matches_cpu() {
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
    let cuda = rollout_sequence_cuda(&observations, &state, &params, &config).unwrap();
    assert_close(&cpu.h.data, &cuda.h.data, 1e-4);
    assert_close(&cpu.r.data, &cuda.r.data, 1e-4);
    assert_close(&cpu.memory.data, &cuda.memory.data, 1e-4);
}

#[cfg(feature = "cuda")]
fn deterministic_values(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| (((idx * 37 + 11) % 23) as f32 - 11.0) * scale)
        .collect()
}

#[cfg(feature = "cuda")]
fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (idx, (a, b)) in left.iter().zip(right).enumerate() {
        assert!(
            (a - b).abs() <= tolerance,
            "idx {idx}: {a} != {b} within {tolerance}"
        );
    }
}
