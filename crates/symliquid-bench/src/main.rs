use std::time::Instant;

use symliquid_core::modules::vsa::VSAMemory;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand_seed(0);
    let hv_dim = 4096;
    let a = VSAMemory::random_bipolar(1, hv_dim, &mut rng);
    let b = VSAMemory::random_bipolar(1, hv_dim, &mut rng);
    #[cfg(feature = "cuda")]
    let loops = 1;
    #[cfg(not(feature = "cuda"))]
    let loops = 2_000;

    let start = Instant::now();
    let out = bind_benchmark(&a, &b, loops)?;
    let elapsed = start.elapsed();
    println!("Benchmark: vsa_bind");
    println!("Backend: {}", backend_label());
    println!("Dim: {hv_dim}");
    println!("Loops: {loops}");
    println!("Runtime ms: {}", elapsed.as_millis());
    println!("Checksum: {:.3}", out.data.iter().take(32).sum::<f32>());
    Ok(())
}

#[cfg(feature = "cuda")]
fn bind_benchmark(
    a: &symliquid_core::Tensor,
    b: &symliquid_core::Tensor,
    _loops: usize,
) -> Result<symliquid_core::Tensor, Box<dyn std::error::Error>> {
    let backend = symliquid_cuda::CudaBackend::new();
    Ok(backend.vsa_bind(a, b)?)
}

#[cfg(not(feature = "cuda"))]
fn bind_benchmark(
    a: &symliquid_core::Tensor,
    b: &symliquid_core::Tensor,
    loops: usize,
) -> Result<symliquid_core::Tensor, Box<dyn std::error::Error>> {
    let mut out = a.clone();
    for _ in 0..loops {
        out = VSAMemory::bind(&out, b)?;
    }
    Ok(out)
}

fn backend_label() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        "cuda-vsa-bind-kernel"
    }
    #[cfg(not(feature = "cuda"))]
    {
        "cpu"
    }
}

fn rand_seed(seed: u64) -> rand::rngs::StdRng {
    use rand::SeedableRng;
    rand::rngs::StdRng::seed_from_u64(seed)
}
