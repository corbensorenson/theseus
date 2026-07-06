use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{json, Value};

use super::*;

pub(super) fn step_duration_summary(
    config: &CodeLmClosureConfig,
    original_train_task_count: usize,
    train_task_count: usize,
    eval_task_count: usize,
    public_task_count: usize,
    estimated_work_steps_before_budget: usize,
    estimated_work_steps_after_budget: usize,
) -> Value {
    json!({
        "policy": "project_theseus_rust_step_duration_v1",
        "duration_control": "work_step_budget_primary_wall_clock_external_safety_fuse_only",
        "epochs": config.epochs.max(1),
        "candidates_per_task": config.candidates_per_task.max(1),
        "max_work_steps": config.max_work_steps,
        "max_work_steps_active": config.max_work_steps > 0,
        "estimated_work_steps_before_budget": estimated_work_steps_before_budget,
        "estimated_work_steps_after_budget": estimated_work_steps_after_budget,
        "estimated_work_steps_within_budget": config.max_work_steps == 0
            || estimated_work_steps_after_budget <= config.max_work_steps,
        "train_rows_trimmed_by_work_budget": original_train_task_count.saturating_sub(train_task_count),
        "train_task_count_after_budget": train_task_count,
        "eval_task_count": eval_task_count,
        "public_task_count": public_task_count,
        "semantic_repeat_accounting": true,
        "score_semantics": "runtime planning metadata only; not learning evidence"
    })
}

pub(super) fn input_file_status(config: &CodeLmClosureConfig) -> Value {
    json!({
        "private_curriculum": file_status(&config.private_curriculum),
        "public_task_manifest": file_status(&config.public_task_manifest),
        "sts_streams": if config.sts_streams.is_empty() {
            json!({"path": "", "exists": false, "bytes": 0})
        } else {
            file_status(&config.sts_streams)
        },
        "score_semantics": "input_integrity_only_not_learning_evidence"
    })
}

pub(super) fn file_status(path: &str) -> Value {
    let item = Path::new(path);
    json!({
        "path": rel(path),
        "exists": item.exists(),
        "bytes": item.metadata().map(|meta| meta.len()).unwrap_or(0),
    })
}

pub(super) fn decoder_contract_task_count(rows: &[CodeTask]) -> usize {
    rows.iter()
        .filter(|task| {
            task.raw
                .get("decoder_contract")
                .and_then(Value::as_object)
                .is_some()
        })
        .count()
}

pub(super) fn decoder_effective_contract_task_count(rows: &[CodeTask]) -> usize {
    rows.iter()
        .filter(|task| {
            decoder_return_shape(task) != "unknown"
                || decoder_type_family(task) != "unknown"
                || !decoder_required_constructs(task).is_empty()
        })
        .count()
}

pub(super) fn record_phase_timing(
    timings: &mut BTreeMap<String, u128>,
    phase: &str,
    started: &mut Instant,
) {
    let elapsed = started.elapsed().as_millis();
    *timings.entry(phase.to_string()).or_insert(0) += elapsed;
    *started = Instant::now();
}

pub(super) fn write_jsonl_binary_sidecar(
    source_path: &Path,
    rows: &[Value],
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut sidecar = source_path.as_os_str().to_os_string();
    sidecar.push(".jsonlb");
    let sidecar_path = PathBuf::from(sidecar);
    if let Some(parent) = sidecar_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(&sidecar_path)?;
    file.write_all(b"THJSONLB1\n")?;
    file.write_all(&(rows.len() as u64).to_le_bytes())?;
    for row in rows {
        let bytes = serde_json::to_vec(row)?;
        let len = u32::try_from(bytes.len())
            .map_err(|_| "candidate row too large for jsonlb sidecar length prefix")?;
        file.write_all(&len.to_le_bytes())?;
        file.write_all(&bytes)?;
    }
    file.flush()?;
    Ok(sidecar_path)
}

pub(super) fn write_progress_report(
    config: &CodeLmClosureConfig,
    started: &Instant,
    stage: &str,
    original_train_task_count: usize,
    train_task_count: usize,
    eval_task_count: usize,
    public_task_count: usize,
    estimated_work_steps_before_budget: usize,
    estimated_work_steps_after_budget: usize,
    progress: Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let report = json!({
        "policy": "project_theseus_code_lm_closure_rust_v1",
        "created_utc": now(),
        "run_status": "in_progress",
        "progress_stage": stage,
        "trigger_state": "YELLOW",
        "checkpoint": rel(&config.checkpoint_out),
        "private_candidate_manifest": rel(&config.private_candidate_out),
        "public_candidate_manifest": rel(&config.public_candidate_out),
        "candidate_generation_heartbeat": rel_path(&candidate_heartbeat_path(config)),
        "summary": {
            "private_train_task_count": train_task_count,
            "private_train_task_count_before_work_budget": original_train_task_count,
            "private_eval_task_count": eval_task_count,
            "public_task_count": public_task_count,
            "input_file_status": input_file_status(config),
            "candidate_generation_heartbeat": rel_path(&candidate_heartbeat_path(config)),
            "step_duration": step_duration_summary(
                config,
                original_train_task_count,
                train_task_count,
                eval_task_count,
                public_task_count,
                estimated_work_steps_before_budget,
                estimated_work_steps_after_budget,
            ),
            "external_inference_calls": 0,
        },
        "progress": progress,
        "runtime_ms": (started.elapsed().as_secs_f64() * 1000.0).round() as u64,
        "external_inference_calls": 0,
    });
    write_json(Path::new(&config.report_out), &report)
}

pub(super) fn candidate_heartbeat_path(config: &CodeLmClosureConfig) -> PathBuf {
    let report = Path::new(&config.report_out);
    let file_name = report
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "code_lm_closure_rust.json".to_string());
    let heartbeat_name = file_name
        .strip_suffix(".json")
        .map(|stem| format!("{stem}.heartbeat.json"))
        .unwrap_or_else(|| format!("{file_name}.heartbeat.json"));
    report
        .parent()
        .map(|parent| parent.join(&heartbeat_name))
        .unwrap_or_else(|| PathBuf::from(heartbeat_name))
}

pub(super) fn rel_path(path: &Path) -> String {
    rel(path.to_string_lossy().as_ref())
}

pub(super) fn candidate_heartbeat_update_every_tasks() -> usize {
    std::env::var("THESEUS_CODE_LM_HEARTBEAT_EVERY_TASKS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

pub(super) fn write_candidate_generation_heartbeat(
    ctx: &CandidateHeartbeat<'_>,
    completed_tasks: usize,
    current_task: Option<&CodeTask>,
    emitted_rows: usize,
    accepted_for_current_task: usize,
    rejected_for_current_task: usize,
    rejection_counts: &BTreeMap<String, usize>,
    status: &str,
) {
    let total_tasks = ctx.total_tasks.max(1);
    let progress_ratio = (completed_tasks as f64 / total_tasks as f64).min(1.0);
    let current = current_task.map(|task| {
        json!({
            "task_id": task.task_id,
            "source_task_id": task.source_task_id,
            "entry_point": task.entry_point,
            "category": task.category,
            "decoder_return_shape": decoder_return_shape(task),
            "decoder_type_family": decoder_type_family(task),
            "required_constructs": decoder_required_constructs(task).into_iter().collect::<Vec<_>>(),
            "plan_hints": semantic_decoder_v2_plan_hints(task, None).into_iter().collect::<Vec<_>>(),
            "benchmark_evidence_level": task.benchmark_evidence_level,
        })
    });
    let payload = json!({
        "policy": "project_theseus_code_lm_candidate_generation_heartbeat_v1",
        "created_utc": now(),
        "run_status": "in_progress",
        "heartbeat_status": status,
        "stage": ctx.stage,
        "phase": ctx.phase,
        "trained": ctx.trained,
        "report_out": rel(&ctx.config.report_out),
        "checkpoint": rel(&ctx.config.checkpoint_out),
        "private_candidate_manifest": rel(&ctx.config.private_candidate_out),
        "public_candidate_manifest": rel(&ctx.config.public_candidate_out),
        "progress": {
            "completed_tasks": completed_tasks,
            "total_tasks": ctx.total_tasks,
            "progress_ratio": (progress_ratio * 1_000_000.0).round() / 1_000_000.0,
            "emitted_rows_so_far": emitted_rows,
            "accepted_for_current_task": accepted_for_current_task,
            "rejected_for_current_task": rejected_for_current_task,
            "rejection_counts_for_current_task": rejection_counts,
            "current_task": current,
        },
        "summary": {
            "private_train_task_count": ctx.train_task_count,
            "private_train_task_count_before_work_budget": ctx.original_train_task_count,
            "private_eval_task_count": ctx.eval_task_count,
            "public_task_count": ctx.public_task_count,
            "step_duration": step_duration_summary(
                ctx.config,
                ctx.original_train_task_count,
                ctx.train_task_count,
                ctx.eval_task_count,
                ctx.public_task_count,
                ctx.estimated_work_steps_before_budget,
                ctx.estimated_work_steps_after_budget,
            ),
            "external_inference_calls": 0,
        },
        "runtime_ms": (ctx.started.elapsed().as_secs_f64() * 1000.0).round() as u64,
        "external_inference_calls": 0,
    });
    let _ = write_json(&candidate_heartbeat_path(ctx.config), &payload);
}
