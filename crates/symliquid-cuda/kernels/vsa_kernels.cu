extern "C" __global__ void vsa_bind_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i];
    }
}

extern "C" __global__ void vsa_bundle_kernel(
    const float* memory,
    const float* x,
    float* out,
    float decay,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = decay * memory[i] + x[i];
    }
}

extern "C" __global__ void vsa_permute_kernel(const float* x, float* out, int n, int shift) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int src = (i - shift) % n;
        if (src < 0) src += n;
        out[i] = x[src];
    }
}
