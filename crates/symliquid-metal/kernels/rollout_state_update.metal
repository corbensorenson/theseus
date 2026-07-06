#include <metal_stdlib>
using namespace metal;

struct RolloutKernelConfig {
    int observation_offset;
    int obs_dim;
    int hidden_dim;
    int reservoir_dim;
    int hv_dim;
    float dt;
    float alpha;
    float memory_decay;
};

struct LinearReadoutConfig {
    int sample_count;
    int input_dim;
    int output_dim;
};

struct ReadoutSgdConfig {
    int sample_start;
    int sample_count;
    int input_dim;
    int output_dim;
    float lr;
};

struct ReadoutBagSgdConfig {
    int sample_start;
    int sample_count;
    int input_dim;
    int output_dim;
    int targets_per_sample;
    float lr;
};

struct StateLinearUpdateConfig {
    int rows;
    int cols;
    float lr;
    float max_abs;
    int has_bias;
};

struct StateHvProjectionUpdateConfig {
    int hv_dim;
    int reservoir_dim;
    float lr;
};

kernel void rollout_state_update_kernel(
    device const float* observations [[buffer(0)]],
    device const float* h_prev [[buffer(1)]],
    device const float* r_prev [[buffer(2)]],
    device const float* memory_prev [[buffer(3)]],
    device float* h_next [[buffer(4)]],
    device float* r_next [[buffer(5)]],
    device float* memory_next [[buffer(6)]],
    device const float* obs_to_h [[buffer(7)]],
    device const float* h_recurrent [[buffer(8)]],
    device const float* h_bias [[buffer(9)]],
    device const float* reservoir_input [[buffer(10)]],
    device const float* reservoir_recurrent [[buffer(11)]],
    device const float* reservoir_bias [[buffer(12)]],
    device const float* hv_proj [[buffer(13)]],
    constant RolloutKernelConfig& cfg [[buffer(14)]],
    uint batch_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    device const float* obs = observations + cfg.observation_offset + batch_index * cfg.obs_dim;
    device const float* h_prev_row = h_prev + batch_index * cfg.hidden_dim;
    device float* h_next_row = h_next + batch_index * cfg.hidden_dim;
    device const float* r_prev_row = r_prev + batch_index * cfg.reservoir_dim;
    device float* r_next_row = r_next + batch_index * cfg.reservoir_dim;
    device const float* memory_prev_row = memory_prev + batch_index * cfg.hv_dim;
    device float* memory_next_row = memory_next + batch_index * cfg.hv_dim;

    for (uint j = tid; j < uint(cfg.hidden_dim); j += threads) {
        float pre = h_bias[j];
        for (int i = 0; i < cfg.obs_dim; ++i) {
            pre += obs_to_h[j * cfg.obs_dim + i] * obs[i];
        }
        for (int k = 0; k < cfg.hidden_dim; ++k) {
            pre += h_recurrent[j * cfg.hidden_dim + k] * h_prev_row[k];
        }
        float candidate = tanh(pre);
        float tau = log(1.0f + exp(pre)) + 1.0e-3f;
        float dh = (-h_prev_row[j] + candidate) / tau;
        h_next_row[j] = h_prev_row[j] + cfg.dt * dh;
    }

    threadgroup_barrier(mem_flags::mem_device);

    for (uint j = tid; j < uint(cfg.reservoir_dim); j += threads) {
        float pre = reservoir_bias[j];
        for (int i = 0; i < cfg.hidden_dim; ++i) {
            pre += reservoir_input[j * cfg.hidden_dim + i] * h_next_row[i];
        }
        for (int k = 0; k < cfg.reservoir_dim; ++k) {
            pre += reservoir_recurrent[j * cfg.reservoir_dim + k] * r_prev_row[k];
        }
        r_next_row[j] = (1.0f - cfg.alpha) * r_prev_row[j] + cfg.alpha * tanh(pre);
    }

    threadgroup_barrier(mem_flags::mem_device);

    for (uint j = tid; j < uint(cfg.hv_dim); j += threads) {
        float dot = 0.0f;
        for (int k = 0; k < cfg.reservoir_dim; ++k) {
            dot += hv_proj[j * cfg.reservoir_dim + k] * r_next_row[k];
        }
        float hv = dot >= 0.0f ? 1.0f : -1.0f;
        memory_next_row[j] = cfg.memory_decay * memory_prev_row[j] + hv;
    }
}

kernel void state_linear_update_kernel(
    device float* weights [[buffer(0)]],
    device float* bias [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device const float* output [[buffer(3)]],
    device const float* target [[buffer(4)]],
    constant StateLinearUpdateConfig& cfg [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    int total = cfg.rows * cfg.cols;
    if (idx >= uint(total) || cfg.rows <= 0 || cfg.cols <= 0) {
        return;
    }

    float norm = 0.0f;
    for (int col = 0; col < cfg.cols; ++col) {
        norm += input[col] * input[col];
    }
    norm = max(norm, 1.0e-6f);

    int row = int(idx) / cfg.cols;
    int col = int(idx) - row * cfg.cols;
    float err = clamp(target[row] - tanh(output[row]), -2.0f, 2.0f);
    float scale = cfg.lr * err / norm;
    float updated = weights[idx] + scale * input[col];
    weights[idx] = clamp(updated, -cfg.max_abs, cfg.max_abs);
    if (cfg.has_bias != 0 && col == 0) {
        bias[row] = clamp(bias[row] + cfg.lr * 0.1f * err, -0.5f, 0.5f);
    }
}

kernel void state_hv_projection_update_kernel(
    device float* hv_proj [[buffer(0)]],
    device const float* reservoir_state [[buffer(1)]],
    device const float* target_hv [[buffer(2)]],
    constant StateHvProjectionUpdateConfig& cfg [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    int total = cfg.hv_dim * cfg.reservoir_dim;
    if (idx >= uint(total) || cfg.hv_dim <= 0 || cfg.reservoir_dim <= 0) {
        return;
    }

    int hv_idx = int(idx) / cfg.reservoir_dim;
    int col = int(idx) - hv_idx * cfg.reservoir_dim;
    device float* row = hv_proj + hv_idx * cfg.reservoir_dim;
    float norm = 0.0f;
    float dot = 0.0f;
    for (int r = 0; r < cfg.reservoir_dim; ++r) {
        norm += reservoir_state[r] * reservoir_state[r];
        dot += row[r] * reservoir_state[r];
    }
    norm = max(norm, 1.0e-6f);

    float pred = dot >= 0.0f ? 1.0f : -1.0f;
    float err = target_hv[hv_idx] - pred;
    if (fabs(err) > 1.0e-7f) {
        row[col] = clamp(row[col] + cfg.lr * err * reservoir_state[col] / norm, -1.0f, 1.0f);
    }
}

kernel void linear_readout_logits_kernel(
    device const float* features [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* logits [[buffer(3)]],
    constant LinearReadoutConfig& cfg [[buffer(4)]],
    uint group_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float partial[256];
    uint sample = group_index / uint(cfg.output_dim);
    uint output = group_index % uint(cfg.output_dim);
    if (sample >= uint(cfg.sample_count) || output >= uint(cfg.output_dim)) {
        return;
    }

    device const float* x = features + sample * cfg.input_dim;
    device const float* w = weights + output * cfg.input_dim;
    float acc = 0.0f;
    for (uint i = tid; i < uint(cfg.input_dim); i += threads) {
        acc += w[i] * x[i];
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        logits[sample * cfg.output_dim + output] = partial[0] + bias[output];
    }
}

kernel void readout_sgd_samples_kernel(
    device const float* features [[buffer(0)]],
    device const int* targets [[buffer(1)]],
    device float* weights [[buffer(2)]],
    device float* bias [[buffer(3)]],
    constant ReadoutSgdConfig& cfg [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float logits[256];
    threadgroup float deltas[256];

    if (cfg.output_dim <= 0 || cfg.output_dim > 256 || cfg.input_dim <= 0) {
        return;
    }

    for (int local = 0; local < cfg.sample_count; ++local) {
        int sample = cfg.sample_start + local;
        device const float* x = features + sample * cfg.input_dim;
        int target = targets[sample];
        if (target < 0 || target >= cfg.output_dim) {
            return;
        }

        for (uint o = tid; o < uint(cfg.output_dim); o += threads) {
            device const float* w = weights + o * cfg.input_dim;
            float acc = bias[o];
            for (int i = 0; i < cfg.input_dim; ++i) {
                acc += w[i] * x[i];
            }
            logits[o] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float max_logit = logits[0];
            for (int o = 1; o < cfg.output_dim; ++o) {
                max_logit = max(max_logit, logits[o]);
            }

            float sum = 0.0f;
            for (int o = 0; o < cfg.output_dim; ++o) {
                float value = exp(logits[o] - max_logit);
                deltas[o] = value;
                sum += value;
            }
            sum = max(sum, 1.0e-12f);

            for (int o = 0; o < cfg.output_dim; ++o) {
                float prob = deltas[o] / sum;
                deltas[o] = prob - (o == target ? 1.0f : 0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int total_weights = cfg.output_dim * cfg.input_dim;
        for (uint idx = tid; idx < uint(total_weights); idx += threads) {
            int o = int(idx) / cfg.input_dim;
            int i = int(idx) - o * cfg.input_dim;
            weights[idx] -= cfg.lr * deltas[o] * x[i];
        }
        for (uint o = tid; o < uint(cfg.output_dim); o += threads) {
            bias[o] -= cfg.lr * deltas[o];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void readout_bag_sgd_samples_kernel(
    device const float* features [[buffer(0)]],
    device const int* target_bags [[buffer(1)]],
    device float* weights [[buffer(2)]],
    device float* bias [[buffer(3)]],
    constant ReadoutBagSgdConfig& cfg [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float logits[256];
    threadgroup float deltas[256];
    threadgroup float target_counts[256];

    if (cfg.output_dim <= 0 || cfg.output_dim > 256 || cfg.input_dim <= 0 || cfg.targets_per_sample <= 0) {
        return;
    }

    for (int local = 0; local < cfg.sample_count; ++local) {
        int sample = cfg.sample_start + local;
        device const float* x = features + sample * cfg.input_dim;
        device const int* bag = target_bags + sample * cfg.targets_per_sample;

        for (uint o = tid; o < uint(cfg.output_dim); o += threads) {
            device const float* w = weights + o * cfg.input_dim;
            float acc = bias[o];
            for (int i = 0; i < cfg.input_dim; ++i) {
                acc += w[i] * x[i];
            }
            logits[o] = acc;
            target_counts[o] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            for (int t = 0; t < cfg.targets_per_sample; ++t) {
                int target = bag[t];
                if (target >= 0 && target < cfg.output_dim) {
                    target_counts[target] += 1.0f / float(cfg.targets_per_sample);
                }
            }

            float max_logit = logits[0];
            for (int o = 1; o < cfg.output_dim; ++o) {
                max_logit = max(max_logit, logits[o]);
            }

            float sum = 0.0f;
            for (int o = 0; o < cfg.output_dim; ++o) {
                float value = exp(logits[o] - max_logit);
                deltas[o] = value;
                sum += value;
            }
            sum = max(sum, 1.0e-12f);

            for (int o = 0; o < cfg.output_dim; ++o) {
                float prob = deltas[o] / sum;
                deltas[o] = prob - target_counts[o];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int total_weights = cfg.output_dim * cfg.input_dim;
        for (uint idx = tid; idx < uint(total_weights); idx += threads) {
            int o = int(idx) / cfg.input_dim;
            int i = int(idx) - o * cfg.input_dim;
            weights[idx] -= cfg.lr * deltas[o] * x[i];
        }
        for (uint o = tid; o < uint(cfg.output_dim); o += threads) {
            bias[o] -= cfg.lr * deltas[o];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
