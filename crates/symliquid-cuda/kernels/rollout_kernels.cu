extern "C" __global__ void rollout_state_update_kernel(
    const float* observations,
    int observation_offset,
    const float* h_prev,
    const float* r_prev,
    const float* memory_prev,
    float* h_next,
    float* r_next,
    float* memory_next,
    const float* obs_to_h,
    const float* h_recurrent,
    const float* h_bias,
    const float* reservoir_input,
    const float* reservoir_recurrent,
    const float* reservoir_bias,
    const float* hv_proj,
    int obs_dim,
    int hidden_dim,
    int reservoir_dim,
    int hv_dim,
    float dt,
    float alpha,
    float memory_decay
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    const float* obs = observations + observation_offset + b * obs_dim;
    const float* h_prev_row = h_prev + b * hidden_dim;
    float* h_next_row = h_next + b * hidden_dim;
    const float* r_prev_row = r_prev + b * reservoir_dim;
    float* r_next_row = r_next + b * reservoir_dim;
    const float* memory_prev_row = memory_prev + b * hv_dim;
    float* memory_next_row = memory_next + b * hv_dim;

    for (int j = tid; j < hidden_dim; j += blockDim.x) {
        float pre = h_bias[j];
        for (int i = 0; i < obs_dim; ++i) {
            pre += obs_to_h[j * obs_dim + i] * obs[i];
        }
        for (int k = 0; k < hidden_dim; ++k) {
            pre += h_recurrent[j * hidden_dim + k] * h_prev_row[k];
        }
        float candidate = tanhf(pre);
        float tau = log1pf(expf(pre)) + 1.0e-3f;
        float dh = (-h_prev_row[j] + candidate) / tau;
        h_next_row[j] = h_prev_row[j] + dt * dh;
    }

    __syncthreads();

    for (int j = tid; j < reservoir_dim; j += blockDim.x) {
        float pre = reservoir_bias[j];
        for (int i = 0; i < hidden_dim; ++i) {
            pre += reservoir_input[j * hidden_dim + i] * h_next_row[i];
        }
        for (int k = 0; k < reservoir_dim; ++k) {
            pre += reservoir_recurrent[j * reservoir_dim + k] * r_prev_row[k];
        }
        r_next_row[j] = (1.0f - alpha) * r_prev_row[j] + alpha * tanhf(pre);
    }

    __syncthreads();

    for (int j = tid; j < hv_dim; j += blockDim.x) {
        float dot = 0.0f;
        for (int k = 0; k < reservoir_dim; ++k) {
            dot += hv_proj[j * reservoir_dim + k] * r_next_row[k];
        }
        float hv = dot >= 0.0f ? 1.0f : -1.0f;
        memory_next_row[j] = memory_decay * memory_prev_row[j] + hv;
    }
}
