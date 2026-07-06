extern "C" __global__ void readout_sgd_samples_kernel(
    const float* features,
    const int* targets,
    float* weights,
    float* bias,
    int sample_start,
    int sample_count,
    int input_dim,
    int output_dim,
    float lr
) {
    __shared__ float logits[256];
    __shared__ float deltas[256];

    if (output_dim > 256) {
        return;
    }

    for (int local = 0; local < sample_count; ++local) {
        int sample = sample_start + local;
        const float* x = features + sample * input_dim;
        int target = targets[sample];

        for (int o = threadIdx.x; o < output_dim; o += blockDim.x) {
            float acc = bias[o];
            const float* w = weights + o * input_dim;
            for (int i = 0; i < input_dim; ++i) {
                acc += w[i] * x[i];
            }
            logits[o] = acc;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float max_logit = logits[0];
            for (int o = 1; o < output_dim; ++o) {
                max_logit = fmaxf(max_logit, logits[o]);
            }

            float sum = 0.0f;
            for (int o = 0; o < output_dim; ++o) {
                float value = expf(logits[o] - max_logit);
                deltas[o] = value;
                sum += value;
            }
            sum = fmaxf(sum, 1.0e-12f);

            for (int o = 0; o < output_dim; ++o) {
                float prob = deltas[o] / sum;
                deltas[o] = prob - (o == target ? 1.0f : 0.0f);
            }
        }
        __syncthreads();

        int total_weights = output_dim * input_dim;
        for (int idx = threadIdx.x; idx < total_weights; idx += blockDim.x) {
            int o = idx / input_dim;
            int i = idx - o * input_dim;
            weights[idx] -= lr * deltas[o] * x[i];
        }
        for (int o = threadIdx.x; o < output_dim; o += blockDim.x) {
            bias[o] -= lr * deltas[o];
        }
        __syncthreads();
    }
}

extern "C" __global__ void readout_bag_sgd_samples_kernel(
    const float* features,
    const int* target_bags,
    float* weights,
    float* bias,
    int sample_start,
    int sample_count,
    int input_dim,
    int output_dim,
    int targets_per_sample,
    float lr
) {
    __shared__ float logits[256];
    __shared__ float deltas[256];
    __shared__ float target_counts[256];

    if (output_dim > 256 || targets_per_sample <= 0) {
        return;
    }

    for (int local = 0; local < sample_count; ++local) {
        int sample = sample_start + local;
        const float* x = features + sample * input_dim;
        const int* bag = target_bags + sample * targets_per_sample;

        for (int o = threadIdx.x; o < output_dim; o += blockDim.x) {
            float acc = bias[o];
            const float* w = weights + o * input_dim;
            for (int i = 0; i < input_dim; ++i) {
                acc += w[i] * x[i];
            }
            logits[o] = acc;
            target_counts[o] = 0.0f;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            for (int t = 0; t < targets_per_sample; ++t) {
                int target = bag[t];
                if (target >= 0 && target < output_dim) {
                    target_counts[target] += 1.0f / (float)targets_per_sample;
                }
            }

            float max_logit = logits[0];
            for (int o = 1; o < output_dim; ++o) {
                max_logit = fmaxf(max_logit, logits[o]);
            }

            float sum = 0.0f;
            for (int o = 0; o < output_dim; ++o) {
                float value = expf(logits[o] - max_logit);
                deltas[o] = value;
                sum += value;
            }
            sum = fmaxf(sum, 1.0e-12f);

            for (int o = 0; o < output_dim; ++o) {
                float prob = deltas[o] / sum;
                deltas[o] = prob - target_counts[o];
            }
        }
        __syncthreads();

        int total_weights = output_dim * input_dim;
        for (int idx = threadIdx.x; idx < total_weights; idx += blockDim.x) {
            int o = idx / input_dim;
            int i = idx - o * input_dim;
            weights[idx] -= lr * deltas[o] * x[i];
        }
        for (int o = threadIdx.x; o < output_dim; o += blockDim.x) {
            bias[o] -= lr * deltas[o];
        }
        __syncthreads();
    }
}

extern "C" __global__ void code_fast_readout_train_kernel(
    const float* features,
    const int* targets,
    const int* salts,
    float* weights,
    float* bias,
    int sample_start,
    int sample_count,
    int input_dim,
    int output_dim,
    float lr
) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_sample = blockIdx.y;
    if (local_sample >= sample_count || feature_idx >= input_dim || output_dim <= 0) {
        return;
    }
    int sample = sample_start + local_sample;

    int target = targets[sample];
    if (target < 0 || target >= output_dim) {
        return;
    }

    int negative = target;
    if (output_dim > 1) {
        int salt = salts[sample];
        int offset = salt % (output_dim - 1);
        if (offset < 0) {
            offset += output_dim - 1;
        }
        negative = (target + 1 + offset) % output_dim;
        if (negative == target) {
            negative = (target + 1) % output_dim;
        }
    }

    float value = features[sample * input_dim + feature_idx];
    if (fabsf(value) >= 1.0e-8f) {
        atomicAdd(&weights[target * input_dim + feature_idx], lr * value);
        if (negative != target) {
            atomicAdd(&weights[negative * input_dim + feature_idx], -0.25f * lr * value);
        }
    }

    if (feature_idx == 0) {
        atomicAdd(&bias[target], lr);
        if (negative != target) {
            atomicAdd(&bias[negative], -0.25f * lr);
        }
    }
}

extern "C" __global__ void binary_readout_score_kernel(
    const float* features,
    const float* weights,
    const float* bias,
    float* scores,
    int sample_count,
    int input_dim,
    int output_dim
) {
    __shared__ float partial[256];

    int sample = blockIdx.x;
    int thread = threadIdx.x;
    if (sample >= sample_count || output_dim < 2) {
        return;
    }

    float acc = 0.0f;
    const float* x = features + sample * input_dim;
    const float* w0 = weights;
    const float* w1 = weights + input_dim;
    for (int i = thread; i < input_dim; i += blockDim.x) {
        acc += (w1[i] - w0[i]) * x[i];
    }
    partial[thread] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            partial[thread] += partial[thread + stride];
        }
        __syncthreads();
    }

    if (thread == 0) {
        scores[sample] = (bias[1] - bias[0]) + partial[0];
    }
}

extern "C" __global__ void weighted_feature_score_kernel(
    const float* features,
    const float* weights,
    float bias,
    float* scores,
    int sample_count,
    int feature_dim
) {
    __shared__ float partial[256];

    int sample = blockIdx.x;
    int thread = threadIdx.x;
    if (sample >= sample_count || feature_dim <= 0 || blockDim.x > 256) {
        return;
    }

    const float* x = features + sample * feature_dim;
    float acc = 0.0f;
    for (int i = thread; i < feature_dim; i += blockDim.x) {
        acc += x[i] * weights[i];
    }
    partial[thread] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            partial[thread] += partial[thread + stride];
        }
        __syncthreads();
    }

    if (thread == 0) {
        scores[sample] = bias + partial[0];
    }
}

extern "C" __global__ void linear_readout_logits_kernel(
    const float* features,
    const float* weights,
    const float* bias,
    float* logits,
    int sample_count,
    int input_dim,
    int output_dim
) {
    __shared__ float partial[256];

    int sample = blockIdx.x;
    int output = blockIdx.y;
    int thread = threadIdx.x;
    if (sample >= sample_count || output >= output_dim) {
        return;
    }

    float acc = 0.0f;
    const float* x = features + sample * input_dim;
    const float* w = weights + output * input_dim;
    for (int i = thread; i < input_dim; i += blockDim.x) {
        acc += w[i] * x[i];
    }
    partial[thread] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            partial[thread] += partial[thread + stride];
        }
        __syncthreads();
    }

    if (thread == 0) {
        logits[sample * output_dim + output] = partial[0] + bias[output];
    }
}

extern "C" __global__ void topk_log_probs_kernel(
    const float* logits,
    int* top_indices,
    float* top_log_probs,
    int sample_count,
    int output_dim,
    int k,
    float top_p
) {
    __shared__ float reduce_values[256];
    __shared__ int reduce_indices[256];
    __shared__ float max_logit_shared;
    __shared__ float denom_shared;
    __shared__ float cumulative_probability;
    __shared__ bool nucleus_closed;

    int sample = blockIdx.x;
    int thread = threadIdx.x;
    if (sample >= sample_count || output_dim <= 0 || k <= 0 || blockDim.x > 256) {
        return;
    }

    const float* row = logits + sample * output_dim;
    float local_max = -3.402823466e+38F;
    for (int i = thread; i < output_dim; i += blockDim.x) {
        local_max = fmaxf(local_max, row[i]);
    }
    reduce_values[thread] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            reduce_values[thread] = fmaxf(reduce_values[thread], reduce_values[thread + stride]);
        }
        __syncthreads();
    }
    if (thread == 0) {
        max_logit_shared = reduce_values[0];
        cumulative_probability = 0.0f;
        nucleus_closed = false;
    }
    __syncthreads();

    float local_denom = 0.0f;
    for (int i = thread; i < output_dim; i += blockDim.x) {
        local_denom += expf(row[i] - max_logit_shared);
    }
    reduce_values[thread] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            reduce_values[thread] += reduce_values[thread + stride];
        }
        __syncthreads();
    }
    if (thread == 0) {
        denom_shared = fmaxf(reduce_values[0], 1.0e-8f);
    }
    __syncthreads();

    top_p = fminf(fmaxf(top_p, 0.0f), 1.0f);
    for (int rank = 0; rank < k; ++rank) {
        if (nucleus_closed) {
            if (thread == 0) {
                top_indices[sample * k + rank] = -1;
                top_log_probs[sample * k + rank] = -3.402823466e+38F;
            }
            __syncthreads();
            continue;
        }

        int local_best_idx = -1;
        float local_best_value = -3.402823466e+38F;
        for (int idx = thread; idx < output_dim; idx += blockDim.x) {
            bool used = false;
            for (int prev = 0; prev < rank; ++prev) {
                if (top_indices[sample * k + prev] == idx) {
                    used = true;
                    break;
                }
            }
            float value = row[idx];
            if (!used && (value > local_best_value || (value == local_best_value && idx < local_best_idx))) {
                local_best_value = value;
                local_best_idx = idx;
            }
        }
        reduce_values[thread] = local_best_value;
        reduce_indices[thread] = local_best_idx;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (thread < stride) {
                float other_value = reduce_values[thread + stride];
                int other_idx = reduce_indices[thread + stride];
                bool other_better = other_value > reduce_values[thread]
                    || (other_value == reduce_values[thread]
                        && other_idx >= 0
                        && (reduce_indices[thread] < 0 || other_idx < reduce_indices[thread]));
                if (other_better) {
                    reduce_values[thread] = other_value;
                    reduce_indices[thread] = other_idx;
                }
            }
            __syncthreads();
        }

        if (thread == 0) {
            int best_idx = reduce_indices[0];
            top_indices[sample * k + rank] = best_idx;
            float log_prob = best_idx >= 0
                ? row[best_idx] - max_logit_shared - logf(denom_shared)
                : -3.402823466e+38F;
            top_log_probs[sample * k + rank] = log_prob;
            if (best_idx >= 0 && top_p < 0.999999f) {
                cumulative_probability += expf(log_prob);
                if (rank > 0 && cumulative_probability >= top_p) {
                    nucleus_closed = true;
                }
            }
        }
        __syncthreads();
    }
}
