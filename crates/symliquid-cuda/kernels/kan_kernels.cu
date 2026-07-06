extern "C" __global__ void kan_rbf_expand_kernel(
    const float* x,
    const float* centers,
    float* out,
    int batch,
    int in_dim,
    int num_basis,
    float sigma
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * in_dim * num_basis;
    if (idx >= total) return;

    int k = idx % num_basis;
    int i = (idx / num_basis) % in_dim;
    int b = idx / (in_dim * num_basis);
    float d = x[b * in_dim + i] - centers[k];
    out[idx] = expf(-(d * d) / (2.0f * sigma * sigma));
}
