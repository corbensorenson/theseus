fn main() {
    for file in [
        "kernels/vsa_kernels.cu",
        "kernels/cleanup_kernels.cu",
        "kernels/reservoir_kernels.cu",
        "kernels/kan_kernels.cu",
        "kernels/liquid_kernels.cu",
        "kernels/fep_kernels.cu",
        "kernels/readout_kernels.cu",
        "kernels/rollout_kernels.cu",
    ] {
        println!("cargo:rerun-if-changed={file}");
    }
}
