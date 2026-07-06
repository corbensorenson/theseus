use crate::cgs::{CgsAccounting, VerificationReport};

#[derive(Debug, Clone)]
pub struct TaskReport {
    pub task: String,
    pub variant: String,
    pub seed: u64,
    pub steps: usize,
    pub accuracy: f32,
    pub loss: f32,
    pub params: usize,
    pub runtime_ms: u128,
    pub residual: f32,
    pub governance_power: f32,
    pub cgs: CgsAccounting,
    pub verification: VerificationReport,
    pub notes: Vec<String>,
}

impl TaskReport {
    pub fn one_line(&self) -> String {
        format!(
            "task={} variant={} seed={} steps={} accuracy={:.3} loss={:.4} residual={:.4} governance_power={:.3} params={} runtime_ms={}",
            self.task,
            self.variant,
            self.seed,
            self.steps,
            self.accuracy,
            self.loss,
            self.residual,
            self.governance_power,
            self.params,
            self.runtime_ms
        )
    }
}

pub fn format_report(report: &TaskReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("Task: {}\n", report.task));
    out.push_str(&format!("Variant: {}\n", report.variant));
    out.push_str(&format!("Seed: {}\n", report.seed));
    out.push_str("Device: cpu\n\n");
    out.push_str(&format!("Steps: {}\n", report.steps));
    out.push_str(&format!("Params: {}\n", report.params));
    out.push_str(&format!("Accuracy: {:.3}\n", report.accuracy));
    out.push_str(&format!("Loss: {:.4}\n", report.loss));
    out.push_str(&format!("Residual: {:.4}\n", report.residual));
    out.push_str(&format!(
        "Governance power: {:.3}\n",
        report.governance_power
    ));
    out.push_str(&format!("Runtime: {} ms\n", report.runtime_ms));
    out.push_str("\nCGS accounting:\n");
    out.push_str(&format!("  Seed cost: {:.1}\n", report.cgs.seed_cost));
    out.push_str(&format!("  Rule cost: {:.1}\n", report.cgs.rule_cost));
    out.push_str(&format!("  Memory cost: {:.1}\n", report.cgs.memory_cost));
    out.push_str(&format!(
        "  Residual cost: {:.4}\n",
        report.cgs.residual_cost
    ));
    out.push_str(&format!(
        "  Verification cost: {:.1}\n",
        report.cgs.verification_cost
    ));
    out.push_str(&format!(
        "  Governance cost: {:.1}\n",
        report.cgs.governance_cost
    ));
    out.push_str(&format!(
        "  Generative leverage: {:.3}\n",
        report.cgs.generative_leverage()
    ));
    out.push_str(&format!(
        "  CGS quality score: {:.6}\n",
        report.cgs.quality_score()
    ));
    out.push_str(&format!(
        "Verification: {}/{} checks passed ({:.3}), {}={:.4}\n",
        report.verification.passed,
        report.verification.checks,
        report.verification.pass_rate(),
        report.verification.residual_name,
        report.verification.residual_value
    ));
    for note in &report.notes {
        out.push_str(&format!("Note: {note}\n"));
    }
    out
}

pub fn format_table(reports: &[TaskReport]) -> String {
    let mut out = String::new();
    out.push_str("Variant              Params      Accuracy    Residual   GovPower   CGS-Q      Runtime(ms)\n");
    out.push_str(
        "---------------------------------------------------------------------------------------\n",
    );
    for report in reports {
        out.push_str(&format!(
            "{:<20} {:>10}  {:>8.3}    {:>7.4}   {:>7.3}   {:>8.6}  {:>10}\n",
            report.variant,
            report.params,
            report.accuracy,
            report.residual,
            report.governance_power,
            report.cgs.quality_score(),
            report.runtime_ms
        ));
    }
    out
}
