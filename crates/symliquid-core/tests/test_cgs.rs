use symliquid_core::cgs::{CgsAccounting, VerificationReport};

#[test]
fn cgs_accounting_exposes_quality_terms() {
    let cgs = CgsAccounting {
        seed_cost: 2.0,
        rule_cost: 8.0,
        memory_cost: 10.0,
        residual_cost: 0.2,
        verification_cost: 5.0,
        governance_cost: 1.0,
        target_cost: 200.0,
        fidelity: 0.9,
        governance_power: 0.4,
    };
    assert!(cgs.total_cost() > 0.0);
    assert!(cgs.generative_leverage() > 1.0);
    assert!(cgs.quality_score().is_finite());

    let verification = VerificationReport::new(10, 8, "toy_residual", 0.2);
    assert_eq!(verification.pass_rate(), 0.8);
}
