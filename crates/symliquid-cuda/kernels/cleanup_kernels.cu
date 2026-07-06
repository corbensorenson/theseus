extern "C" __global__ void cleanup_similarity_kernel(
    const float* query,
    const float* table,
    float* scores,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float partial[];

    float acc = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        acc += query[col] * table[row * cols + col];
    }
    partial[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0 && row < rows) {
        scores[row] = partial[0];
    }
}
