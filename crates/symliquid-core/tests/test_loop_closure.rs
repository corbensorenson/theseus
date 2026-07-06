use std::collections::{BTreeMap, BTreeSet};

use symliquid_core::loop_closure::{
    detect_loop_candidates, infer_parameter_states, synthesize_tool_card, ExecutionMode,
    LatencyClass, ParameterState, RiskTier, RouteRequest, RuntimeTier, ToolRegistry,
    TrajectoryRecord, VerificationGrade,
};

#[test]
fn detects_repeated_verified_trajectories() {
    let trajectories = vec![
        trajectory("t1", "repo_test", "crate=symliquid-core", true),
        trajectory("t2", "repo_test", "crate=symliquid-cuda", true),
        trajectory("t3", "repo_test", "crate=symliquid-cli", true),
        trajectory("t4", "whitepaper", "topic=cgs", true),
        trajectory("t5", "repo_test", "crate=broken", false),
    ];

    let candidates = detect_loop_candidates(&trajectories, 3);
    assert_eq!(candidates.len(), 1);
    let candidate = &candidates[0];
    assert_eq!(candidate.task_family, "repo_test");
    assert_eq!(candidate.recurrence_count, 3);
    assert_eq!(
        candidate.invariant_steps,
        vec!["inspect_repo", "run_tests", "parse_failures", "report"]
    );
    assert_eq!(candidate.parameter_keys, vec!["crate"]);

    let states = infer_parameter_states(&trajectories[..3]);
    assert_eq!(states.get("crate"), Some(&ParameterState::Parameter));
}

#[test]
fn registry_routes_only_when_parameters_and_risk_match() {
    let trajectories = vec![
        trajectory("t1", "repo_test", "crate=symliquid-core", true),
        trajectory("t2", "repo_test", "crate=symliquid-cuda", true),
    ];
    let candidate = detect_loop_candidates(&trajectories, 2).remove(0);
    let mut tool = synthesize_tool_card(&candidate, "run_repo_tests");
    tool.risk_tier = RiskTier::Medium;

    let mut registry = ToolRegistry::default();
    registry.register(tool);

    let params = BTreeSet::from(["crate".to_string()]);
    assert!(registry
        .route("repo_test", &params, RiskTier::Medium)
        .is_some());
    assert!(registry
        .route("repo_test", &params, RiskTier::Low)
        .is_none());
    assert!(registry
        .route("repo_test", &BTreeSet::new(), RiskTier::Medium)
        .is_none());
}

#[test]
fn rich_router_selects_interpreter_tool_or_reflex_mode() {
    let trajectories = vec![
        trajectory("t1", "repo_test", "crate=symliquid-core", true),
        trajectory("t2", "repo_test", "crate=symliquid-cuda", true),
    ];
    let candidate = detect_loop_candidates(&trajectories, 2).remove(0);
    let mut tool = synthesize_tool_card(&candidate, "run_repo_tests");
    tool.risk_tier = RiskTier::Medium;
    tool.verification_grade = VerificationGrade::SyntheticPassed;
    tool.latency_class = LatencyClass::Interactive;

    let mut reflex = tool.clone();
    reflex.name = "emergency_hold".to_string();
    reflex.task_family = "deployment_hold".to_string();
    reflex.parameters = vec!["service".to_string()];
    reflex.risk_tier = RiskTier::Critical;
    reflex.verification_grade = VerificationGrade::Certified;
    reflex.runtime_tier = RuntimeTier::RealtimeReflexRuntime;
    reflex.latency_class = LatencyClass::SafetyCriticalReflex;
    reflex.fallback_mode = ExecutionMode::ReflexFailsafe;

    let mut registry = ToolRegistry::default();
    registry.register(tool);
    registry.register(reflex);

    let compiled = registry.route_execution(&RouteRequest {
        task_family: "repo_test".to_string(),
        available_parameters: BTreeSet::from(["crate".to_string()]),
        max_risk: RiskTier::Medium,
        min_verification_grade: VerificationGrade::HeldOutPassed,
        latency_class: LatencyClass::Interactive,
        safety_critical: false,
        environment_verified: true,
    });
    assert_eq!(compiled.mode, ExecutionMode::CompiledTool);
    assert_eq!(compiled.tool_name.as_deref(), Some("run_repo_tests"));

    let interpreter = registry.route_execution(&RouteRequest {
        task_family: "repo_test".to_string(),
        available_parameters: BTreeSet::from(["crate".to_string()]),
        max_risk: RiskTier::Medium,
        min_verification_grade: VerificationGrade::HeldOutPassed,
        latency_class: LatencyClass::Interactive,
        safety_critical: false,
        environment_verified: false,
    });
    assert_eq!(interpreter.mode, ExecutionMode::Interpreter);

    let reflex_decision = registry.route_execution(&RouteRequest {
        task_family: "deployment_hold".to_string(),
        available_parameters: BTreeSet::from(["service".to_string()]),
        max_risk: RiskTier::Critical,
        min_verification_grade: VerificationGrade::Certified,
        latency_class: LatencyClass::SafetyCriticalReflex,
        safety_critical: true,
        environment_verified: true,
    });
    assert_eq!(reflex_decision.mode, ExecutionMode::ReflexFailsafe);
    assert_eq!(reflex_decision.tool_name.as_deref(), Some("emergency_hold"));
}

fn trajectory(
    id: &str,
    task_family: &str,
    parameter: &str,
    verification_passed: bool,
) -> TrajectoryRecord {
    let (key, value) = parameter.split_once('=').unwrap();
    let mut parameters = BTreeMap::new();
    parameters.insert(key.to_string(), value.to_string());
    TrajectoryRecord {
        id: id.to_string(),
        task_family: task_family.to_string(),
        intent: "repeatable workflow".to_string(),
        steps: vec![
            "inspect_repo".to_string(),
            "run_tests".to_string(),
            "parse_failures".to_string(),
            "report".to_string(),
        ],
        parameters,
        verification_passed,
        residual: if verification_passed { 0.0 } else { 1.0 },
        cost_units: 10.0,
        environment: BTreeMap::from([("package_manager".to_string(), "cargo".to_string())]),
        risk_tier: RiskTier::Medium,
        latency_class: LatencyClass::Interactive,
        safety_event: false,
    }
}
