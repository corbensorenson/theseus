use rand::SeedableRng;
use symliquid_core::modules::kan_lite::KANLiteLayer;
use symliquid_core::Tensor;

#[test]
fn kan_forward_shape_and_regularization() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let layer = KANLiteLayer::new(3, 5, 7, &mut rng).unwrap();
    let x = Tensor::ones(4, 3);
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.shape(), (4, 5));
    assert!(layer.regularization(1e-3, 1e-3).is_finite());
    assert!(layer.edge_value(0, 0, 0.25).unwrap().is_finite());
}
