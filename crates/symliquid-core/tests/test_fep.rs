use rand::SeedableRng;
use symliquid_core::modules::fep::FEPLayer;

#[test]
fn fep_scores_all_candidate_actions() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(4);
    let layer = FEPLayer::new(3, 3, 4, &mut rng).unwrap();
    let scores = layer
        .expected_free_energy(&[0.7, 0.2, 0.1], &[0.05, 0.9, 0.05])
        .unwrap();
    assert_eq!(scores.len(), 4);
    assert!(scores.iter().all(|v| v.is_finite()));
}

#[test]
fn fep_belief_update_prefers_likely_observation() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(5);
    let layer = FEPLayer::new(3, 3, 2, &mut rng).unwrap();
    let posterior = layer.update_belief(&[1.0 / 3.0; 3], 0, 1).unwrap();
    assert_eq!(posterior.len(), 3);
    assert!(posterior[1] > posterior[0]);
}
