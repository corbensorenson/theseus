use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum RiskTier {
    #[default]
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum VerificationGrade {
    #[default]
    Unverified,
    ReplayPassed,
    HeldOutPassed,
    SyntheticPassed,
    AdversarialTested,
    RuntimeMonitored,
    HumanApproved,
    Certified,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeTier {
    TextTemplate,
    #[default]
    StructuredWorkflow,
    TypedFunction,
    SandboxedRuntime,
    MemorySafeRuntime,
    RealtimeReflexRuntime,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum LatencyClass {
    NoUrgency,
    #[default]
    Interactive,
    Operational,
    Realtime,
    SafetyCriticalReflex,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    #[default]
    Interpreter,
    CompiledTool,
    ReflexFailsafe,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParameterState {
    Invariant,
    Parameter,
    Precondition,
    UnknownAssumption,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryRecord {
    pub id: String,
    pub task_family: String,
    pub intent: String,
    pub steps: Vec<String>,
    pub parameters: BTreeMap<String, String>,
    pub verification_passed: bool,
    pub residual: f32,
    pub cost_units: f32,
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    #[serde(default)]
    pub risk_tier: RiskTier,
    #[serde(default)]
    pub latency_class: LatencyClass,
    #[serde(default)]
    pub safety_event: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopCandidate {
    pub task_family: String,
    pub trajectory_ids: Vec<String>,
    pub invariant_steps: Vec<String>,
    pub parameter_keys: Vec<String>,
    pub recurrence_count: usize,
    pub mean_residual: f32,
    pub estimated_savings: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCard {
    pub name: String,
    pub version: String,
    pub purpose: String,
    pub task_family: String,
    pub input_schema: String,
    pub output_schema: String,
    pub parameters: Vec<String>,
    pub preconditions: Vec<String>,
    pub postconditions: Vec<String>,
    pub verification_tests: Vec<String>,
    #[serde(default)]
    pub verification_grade: VerificationGrade,
    #[serde(default)]
    pub active_probes: Vec<String>,
    #[serde(default)]
    pub hidden_assumptions: Vec<String>,
    pub known_edge_cases: Vec<String>,
    pub failure_modes: Vec<String>,
    pub risk_tier: RiskTier,
    #[serde(default)]
    pub runtime_tier: RuntimeTier,
    #[serde(default)]
    pub latency_class: LatencyClass,
    #[serde(default)]
    pub allowed_side_effects: Vec<String>,
    #[serde(default)]
    pub permissions: Vec<String>,
    #[serde(default)]
    pub runtime_monitors: Vec<String>,
    pub fallback: String,
    #[serde(default)]
    pub fallback_mode: ExecutionMode,
    pub provenance: Vec<String>,
    pub success_rate: f32,
    pub estimated_savings: f32,
    pub retirement_criteria: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolRegistry {
    pub tools: Vec<ToolCard>,
}

#[derive(Debug, Clone)]
pub struct RouteRequest {
    pub task_family: String,
    pub available_parameters: BTreeSet<String>,
    pub max_risk: RiskTier,
    pub min_verification_grade: VerificationGrade,
    pub latency_class: LatencyClass,
    pub safety_critical: bool,
    pub environment_verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouteDecision {
    pub mode: ExecutionMode,
    pub tool_name: Option<String>,
    pub reason: String,
}

impl ToolRegistry {
    pub fn register(&mut self, tool: ToolCard) {
        if let Some(existing) = self
            .tools
            .iter_mut()
            .find(|existing| existing.name == tool.name && existing.version == tool.version)
        {
            *existing = tool;
        } else {
            self.tools.push(tool);
        }
    }

    pub fn route(
        &self,
        task_family: &str,
        available_parameters: &BTreeSet<String>,
        max_risk: RiskTier,
    ) -> Option<&ToolCard> {
        self.tools
            .iter()
            .filter(|tool| tool.task_family == task_family)
            .filter(|tool| tool.risk_tier <= max_risk)
            .filter(|tool| {
                tool.parameters
                    .iter()
                    .all(|parameter| available_parameters.contains(parameter))
            })
            .max_by(|left, right| {
                closure_score(left)
                    .partial_cmp(&closure_score(right))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    pub fn route_execution(&self, request: &RouteRequest) -> RouteDecision {
        if request.safety_critical || request.latency_class == LatencyClass::SafetyCriticalReflex {
            if let Some(tool) = self.best_tool(request, true) {
                return RouteDecision {
                    mode: ExecutionMode::ReflexFailsafe,
                    tool_name: Some(tool.name.clone()),
                    reason: "safety-critical request matched a reflex/failsafe tool".to_string(),
                };
            }
            return RouteDecision {
                mode: ExecutionMode::ReflexFailsafe,
                tool_name: None,
                reason: "safety-critical request has no certified reflex tool; trigger failsafe"
                    .to_string(),
            };
        }

        if !request.environment_verified {
            return RouteDecision {
                mode: ExecutionMode::Interpreter,
                tool_name: None,
                reason: "environment preconditions are not verified".to_string(),
            };
        }

        if let Some(tool) = self.best_tool(request, false) {
            return RouteDecision {
                mode: ExecutionMode::CompiledTool,
                tool_name: Some(tool.name.clone()),
                reason: "matched verified closed-loop tool".to_string(),
            };
        }

        RouteDecision {
            mode: ExecutionMode::Interpreter,
            tool_name: None,
            reason: "no closed-loop tool satisfied parameters, risk, latency, and verification"
                .to_string(),
        }
    }

    fn best_tool(&self, request: &RouteRequest, require_reflex: bool) -> Option<&ToolCard> {
        self.tools
            .iter()
            .filter(|tool| tool.task_family == request.task_family)
            .filter(|tool| tool.risk_tier <= request.max_risk)
            .filter(|tool| tool.verification_grade >= request.min_verification_grade)
            .filter(|tool| tool.latency_class >= request.latency_class)
            .filter(|tool| {
                !require_reflex || tool.runtime_tier == RuntimeTier::RealtimeReflexRuntime
            })
            .filter(|tool| {
                tool.parameters
                    .iter()
                    .all(|parameter| request.available_parameters.contains(parameter))
            })
            .max_by(|left, right| {
                closure_score(left)
                    .partial_cmp(&closure_score(right))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

pub fn detect_loop_candidates(
    trajectories: &[TrajectoryRecord],
    min_recurrence: usize,
) -> Vec<LoopCandidate> {
    let mut groups: BTreeMap<String, Vec<&TrajectoryRecord>> = BTreeMap::new();
    for trajectory in trajectories
        .iter()
        .filter(|trajectory| trajectory.verification_passed)
    {
        groups
            .entry(loop_signature(trajectory))
            .or_default()
            .push(trajectory);
    }

    groups
        .into_values()
        .filter(|group| group.len() >= min_recurrence.max(2))
        .map(|group| {
            let task_family = group[0].task_family.clone();
            let invariant_steps = common_prefix_steps(&group);
            let parameter_keys = parameter_keys(&group);
            let recurrence_count = group.len();
            let mean_residual = group
                .iter()
                .map(|trajectory| trajectory.residual)
                .sum::<f32>()
                / recurrence_count as f32;
            let mean_cost = group
                .iter()
                .map(|trajectory| trajectory.cost_units)
                .sum::<f32>()
                / recurrence_count as f32;
            LoopCandidate {
                task_family,
                trajectory_ids: group
                    .iter()
                    .map(|trajectory| trajectory.id.clone())
                    .collect(),
                invariant_steps,
                parameter_keys,
                recurrence_count,
                mean_residual,
                estimated_savings: mean_cost * recurrence_count.saturating_sub(1) as f32,
            }
        })
        .collect()
}

pub fn synthesize_tool_card(candidate: &LoopCandidate, name: impl Into<String>) -> ToolCard {
    let success_rate = (1.0 - candidate.mean_residual).clamp(0.0, 1.0);
    ToolCard {
        name: name.into(),
        version: "0.1.0".to_string(),
        purpose: format!(
            "Closed loop tool for repeated {} trajectories.",
            candidate.task_family
        ),
        task_family: candidate.task_family.clone(),
        input_schema: "typed_parameters".to_string(),
        output_schema: "verified_artifact_or_action".to_string(),
        parameters: candidate.parameter_keys.clone(),
        preconditions: vec![
            "task_family_matches".to_string(),
            "required_parameters_available".to_string(),
            "verification_available".to_string(),
        ],
        postconditions: vec!["verification_passed_or_fallback_triggered".to_string()],
        verification_tests: candidate
            .trajectory_ids
            .iter()
            .map(|id| format!("replay_trajectory:{id}"))
            .collect(),
        verification_grade: VerificationGrade::ReplayPassed,
        active_probes: active_parameter_probe_plan(candidate),
        hidden_assumptions: vec!["environment_may_differ_from_source_trajectories".to_string()],
        known_edge_cases: vec!["novel_parameters_require_reasoning_fallback".to_string()],
        failure_modes: vec![
            "precondition_miss".to_string(),
            "stale_environment_assumption".to_string(),
            "verifier_false_positive".to_string(),
        ],
        risk_tier: RiskTier::Medium,
        runtime_tier: RuntimeTier::TypedFunction,
        latency_class: LatencyClass::Interactive,
        allowed_side_effects: vec!["declared_output_only".to_string()],
        permissions: vec!["least_privilege_required".to_string()],
        runtime_monitors: vec!["postcondition_verifier".to_string()],
        fallback: "fresh_reasoning_with_logging".to_string(),
        fallback_mode: ExecutionMode::Interpreter,
        provenance: candidate.trajectory_ids.clone(),
        success_rate,
        estimated_savings: candidate.estimated_savings,
        retirement_criteria: vec![
            "success_rate_below_threshold".to_string(),
            "environment_schema_changed".to_string(),
        ],
    }
}

pub fn infer_parameter_states(
    trajectories: &[TrajectoryRecord],
) -> BTreeMap<String, ParameterState> {
    let mut values_by_key: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut environments_by_key: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for trajectory in trajectories {
        for (key, value) in &trajectory.parameters {
            values_by_key
                .entry(key.clone())
                .or_default()
                .insert(value.clone());
        }
        for (key, value) in &trajectory.environment {
            environments_by_key
                .entry(key.clone())
                .or_default()
                .insert(value.clone());
        }
    }

    let mut states = BTreeMap::new();
    for (key, values) in values_by_key {
        let state = if values.len() > 1 {
            ParameterState::Parameter
        } else {
            ParameterState::UnknownAssumption
        };
        states.insert(key, state);
    }
    for (key, values) in environments_by_key {
        let state = if values.len() > 1 {
            ParameterState::Precondition
        } else {
            ParameterState::UnknownAssumption
        };
        states.entry(format!("env:{key}")).or_insert(state);
    }
    states
}

pub fn active_parameter_probe_plan(candidate: &LoopCandidate) -> Vec<String> {
    let mut probes = Vec::new();
    for parameter in &candidate.parameter_keys {
        probes.push(format!("counterfactual_replay:vary:{parameter}"));
        probes.push(format!("synthetic_case:missing_or_extreme:{parameter}"));
    }
    probes.extend([
        "environment_interrogation:runtime_versions".to_string(),
        "adversarial_case:ambiguous_or_partial_inputs".to_string(),
        "sandbox_probe:permission_and_side_effect_boundary".to_string(),
    ]);
    probes
}

fn closure_score(tool: &ToolCard) -> f32 {
    let risk_penalty = match tool.risk_tier {
        RiskTier::Low => 0.0,
        RiskTier::Medium => 0.1,
        RiskTier::High => 0.35,
        RiskTier::Critical => 0.75,
    };
    let verification_bonus = match tool.verification_grade {
        VerificationGrade::Unverified => -0.5,
        VerificationGrade::ReplayPassed => 0.0,
        VerificationGrade::HeldOutPassed => 0.1,
        VerificationGrade::SyntheticPassed => 0.15,
        VerificationGrade::AdversarialTested => 0.2,
        VerificationGrade::RuntimeMonitored => 0.25,
        VerificationGrade::HumanApproved => 0.3,
        VerificationGrade::Certified => 0.45,
    };
    tool.success_rate + tool.estimated_savings.ln_1p() * 0.05 + verification_bonus - risk_penalty
}

fn loop_signature(trajectory: &TrajectoryRecord) -> String {
    format!("{}::{}", trajectory.task_family, trajectory.steps.join(">"))
}

fn common_prefix_steps(group: &[&TrajectoryRecord]) -> Vec<String> {
    let Some(first) = group.first() else {
        return Vec::new();
    };
    let mut prefix = Vec::new();
    for (idx, step) in first.steps.iter().enumerate() {
        if group
            .iter()
            .all(|trajectory| trajectory.steps.get(idx) == Some(step))
        {
            prefix.push(step.clone());
        } else {
            break;
        }
    }
    prefix
}

fn parameter_keys(group: &[&TrajectoryRecord]) -> Vec<String> {
    let mut keys = BTreeSet::new();
    for trajectory in group {
        keys.extend(trajectory.parameters.keys().cloned());
    }
    keys.into_iter().collect()
}
