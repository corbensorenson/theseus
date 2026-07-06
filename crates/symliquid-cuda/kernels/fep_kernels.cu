extern "C" __global__ void efe_score_kernel(
    const float* risk,
    const float* ambiguity,
    const float* info_gain,
    float* efe,
    int action_dim,
    float epistemic_weight
) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < action_dim) {
        efe[a] = risk[a] + ambiguity[a] - epistemic_weight * info_gain[a];
    }
}
