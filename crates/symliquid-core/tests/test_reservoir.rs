use rand::SeedableRng;
use symliquid_core::modules::reservoir::Reservoir;
use symliquid_core::Tensor;

#[test]
fn reservoir_is_stable_and_shape_correct() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let reservoir = Reservoir::new(8, 32, 0.9, 0.6, false, &mut rng).unwrap();
    let radius = reservoir.spectral_radius_estimate();
    assert!(radius.is_finite());
    assert!((radius - 0.9).abs() < 0.35, "radius was {radius}");
    let h = Tensor::ones(3, 8);
    let r = Tensor::zeros(3, 32);
    let next = reservoir.forward(&h, &r).unwrap();
    assert_eq!(next.shape(), (3, 32));
    assert!(next.data.iter().all(|v| v.abs() <= 1.0));
}
