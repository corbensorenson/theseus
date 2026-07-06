#[derive(Debug, Clone)]
pub struct CgsAccounting {
    pub seed_cost: f32,
    pub rule_cost: f32,
    pub memory_cost: f32,
    pub residual_cost: f32,
    pub verification_cost: f32,
    pub governance_cost: f32,
    pub target_cost: f32,
    pub fidelity: f32,
    pub governance_power: f32,
}

impl CgsAccounting {
    pub fn total_cost(&self) -> f32 {
        self.seed_cost
            + self.rule_cost
            + self.memory_cost
            + self.residual_cost
            + self.verification_cost
            + self.governance_cost
    }

    pub fn generative_leverage(&self) -> f32 {
        let structural_cost = self.seed_cost + self.rule_cost + self.memory_cost;
        self.target_cost / structural_cost.max(1.0)
    }

    pub fn quality_score(&self) -> f32 {
        let numerator = self.governance_power.max(0.0)
            * self.fidelity.max(0.0)
            * self.generative_leverage().max(0.0);
        let denominator = 1.0 + self.residual_cost + self.verification_cost + self.governance_cost;
        numerator / denominator.max(1e-6)
    }
}

impl Default for CgsAccounting {
    fn default() -> Self {
        Self {
            seed_cost: 0.0,
            rule_cost: 0.0,
            memory_cost: 0.0,
            residual_cost: 0.0,
            verification_cost: 0.0,
            governance_cost: 0.0,
            target_cost: 0.0,
            fidelity: 0.0,
            governance_power: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub checks: usize,
    pub passed: usize,
    pub residual_name: String,
    pub residual_value: f32,
}

impl VerificationReport {
    pub fn new(
        checks: usize,
        passed: usize,
        residual_name: impl Into<String>,
        residual_value: f32,
    ) -> Self {
        Self {
            checks,
            passed,
            residual_name: residual_name.into(),
            residual_value,
        }
    }

    pub fn pass_rate(&self) -> f32 {
        self.passed as f32 / self.checks.max(1) as f32
    }
}

impl Default for VerificationReport {
    fn default() -> Self {
        Self {
            checks: 0,
            passed: 0,
            residual_name: "residual".to_string(),
            residual_value: 0.0,
        }
    }
}
