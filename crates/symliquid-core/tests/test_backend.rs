use rand::SeedableRng;
use symliquid_core::backend::{Backend, CpuBackend};
use symliquid_core::modules::vsa::VSAMemory;

#[test]
fn cpu_backend_matches_vsa_primitives() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    let backend = CpuBackend;
    let a = VSAMemory::random_bipolar(1, 64, &mut rng);
    let b = VSAMemory::random_bipolar(1, 64, &mut rng);
    let bound = backend.vsa_bind(&a, &b).unwrap();
    let recovered = backend.vsa_bind(&bound, &a).unwrap();
    assert_eq!(recovered, b);

    let efe = backend
        .expected_free_energy(&[1.0, 0.5], &[0.1, 0.2], &[0.4, 0.1], 0.5)
        .unwrap();
    assert_eq!(efe.len(), 2);
    assert!(efe[1] < efe[0]);
}
