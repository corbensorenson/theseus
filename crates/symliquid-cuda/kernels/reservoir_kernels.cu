extern "C" __global__ void reservoir_update_kernel(
    const float* r_prev,
    const float* h,
    const float* recurrent,
    const float* input,
    const float* bias,
    float* r_next,
    int batch,
    int hidden_dim,
    int reservoir_dim,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * reservoir_dim;
    if (idx >= total) return;

    int b = idx / reservoir_dim;
    int j = idx % reservoir_dim;
    float pre = bias[j];
    for (int k = 0; k < reservoir_dim; ++k) {
        pre += recurrent[j * reservoir_dim + k] * r_prev[b * reservoir_dim + k];
    }
    for (int i = 0; i < hidden_dim; ++i) {
        pre += input[j * hidden_dim + i] * h[b * hidden_dim + i];
    }
    r_next[idx] = (1.0f - alpha) * r_prev[idx] + alpha * tanhf(pre);
}
