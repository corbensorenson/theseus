extern "C" __global__ void liquid_elementwise_update_kernel(
    const float* h_prev,
    const float* candidate,
    const float* tau_pre,
    float* h_next,
    int n,
    float dt,
    float eps
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float tau = log1pf(expf(tau_pre[i])) + eps;
    float dh = (-h_prev[i] + tanhf(candidate[i])) / tau;
    h_next[i] = h_prev[i] + dt * dh;
}
