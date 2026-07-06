use rand::SeedableRng;
use symliquid_core::modules::liquid::LiquidCell;
use symliquid_core::Tensor;

#[test]
fn liquid_outputs_are_finite() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let cell = LiquidCell::new(4, 6, &mut rng).unwrap();
    let x = Tensor::ones(2, 4);
    let h = Tensor::zeros(2, 6);
    let next = cell.forward(&x, &h).unwrap();
    assert_eq!(next.shape(), (2, 6));
    assert!(next.data.iter().all(|v| v.is_finite()));
}
