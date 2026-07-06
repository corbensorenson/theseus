use rand::SeedableRng;
use symliquid_core::modules::vsa::VSAMemory;
use symliquid_core::Tensor;

#[test]
fn vsa_binding_unbinding_recovers_filler() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(2);
    let role = VSAMemory::random_bipolar(1, 256, &mut rng);
    let filler = VSAMemory::random_bipolar(1, 256, &mut rng);
    let bound = VSAMemory::bind(&role, &filler).unwrap();
    let recovered = VSAMemory::unbind(&bound, &role).unwrap();
    assert_eq!(recovered, filler);
    assert!(VSAMemory::consistency_loss(&role, &filler).unwrap() < 1e-6);
}

#[test]
fn vsa_cleanup_selects_nearest_symbol() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    let table = VSAMemory::random_bipolar(8, 128, &mut rng);
    let query = Tensor::new(1, 128, table.row(5).to_vec()).unwrap();
    let (idx, score) = VSAMemory::cleanup(&query, &table).unwrap();
    assert_eq!(idx, 5);
    assert!(score > 0.99);
}
