use symliquid_core::{SymLiquidConfig, SymLiquidFEPNet, Tensor};

#[test]
fn integrated_model_forward_and_readout_update() {
    let cfg = SymLiquidConfig {
        input_dim: 4,
        hidden_dim: 8,
        reservoir_dim: 16,
        hv_dim: 32,
        output_dim: 3,
        latent_dim: 3,
        obs_dim: 3,
        action_dim: 2,
        ..SymLiquidConfig::default()
    };
    let mut model = SymLiquidFEPNet::new(cfg, 0).unwrap();
    let x = Tensor::ones(2, 4);
    let out = model
        .forward_step(&x, None, None, Some(&[0.1, 0.8, 0.1]))
        .unwrap();
    assert_eq!(out.logits.shape(), (2, 3));
    assert_eq!(out.memory.shape(), (2, 32));
    assert!(out.expected_free_energy.is_some());
    assert!(out.residual.is_finite());
    let (loss, _acc) = model.train_readout_sgd(&out.memory, &[1, 2], 0.01).unwrap();
    assert!(loss.is_finite());
}
