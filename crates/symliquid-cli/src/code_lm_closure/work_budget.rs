use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde_json::{json, Value};

use super::*;

pub(super) fn estimated_rust_work_steps(
    train_rows: &[CodeTask],
    eval_rows: &[CodeTask],
    public_rows: &[CodeTask],
    epochs: usize,
    candidates_per_task: usize,
) -> usize {
    let train_token_steps = train_rows
        .iter()
        .map(|task| task_train_step_cost(task, epochs))
        .sum::<usize>();
    let candidate_steps = eval_rows
        .len()
        .saturating_add(public_rows.len())
        .saturating_mul(candidates_per_task)
        .saturating_mul(4);
    train_token_steps.saturating_add(candidate_steps)
}

pub(super) fn trim_train_rows_to_work_budget(
    train_rows: &[CodeTask],
    eval_rows: &[CodeTask],
    public_rows: &[CodeTask],
    epochs: usize,
    candidates_per_task: usize,
    max_work_steps: usize,
) -> Vec<CodeTask> {
    if max_work_steps == 0 {
        return train_rows.to_vec();
    }
    let fixed_eval_candidate_steps = eval_rows
        .len()
        .saturating_add(public_rows.len())
        .saturating_mul(candidates_per_task)
        .saturating_mul(4);
    let train_budget = max_work_steps.saturating_sub(fixed_eval_candidate_steps);
    let mut used = 0usize;
    let mut out = Vec::new();
    if !stratified_work_budget_admission_enabled() {
        let ordered = if execution_shape_category_stratification_needed(train_rows) {
            category_stratified_work_budget_order(train_rows)
        } else {
            train_rows.iter().collect::<Vec<_>>()
        };
        for task in ordered {
            let cost = task_train_step_cost(task, epochs);
            if !out.is_empty() && used.saturating_add(cost) > train_budget {
                break;
            }
            used = used.saturating_add(cost);
            out.push(task.clone());
        }
        let admitted = if out.is_empty() {
            train_rows.iter().take(1).cloned().collect()
        } else {
            out
        };
        return rescue_starved_target_families(train_rows, admitted, epochs, train_budget);
    }
    for task in stratified_work_budget_order(train_rows, epochs) {
        let cost = task_train_step_cost(task, epochs);
        if used.saturating_add(cost) > train_budget {
            continue;
        }
        used = used.saturating_add(cost);
        out.push(task.clone());
    }
    if out.is_empty() {
        train_rows
            .iter()
            .min_by_key(|task| task_train_step_cost(task, epochs))
            .cloned()
            .into_iter()
            .collect()
    } else {
        out
    }
}

pub(super) fn work_budget_admission_summary(
    before: &[CodeTask],
    after: &[CodeTask],
    max_work_steps: usize,
    estimated_before: usize,
    estimated_after: usize,
) -> Value {
    let before_counts = admission_family_counts(before);
    let after_counts = admission_family_counts(after);
    let target_families = [
        "execution_shape",
        "algorithm_choice",
        "branch_loop_local_interface",
        "return_shape_interface",
        "edge_conditions",
        "type_handling",
        "parsing",
        "recurrence_state",
    ];
    let min_rows = target_family_starvation_rescue_min_rows();
    let undercovered = target_families
        .iter()
        .filter_map(|family| {
            let pool = *before_counts.get(*family).unwrap_or(&0usize);
            let admitted = *after_counts.get(*family).unwrap_or(&0usize);
            let target = min_rows.min(pool);
            if pool > 0 && admitted < target {
                Some(
                    json!({"family": family, "pool": pool, "admitted": admitted, "target": target}),
                )
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let undercovered_target_family_count = undercovered.len();
    let stratified_enabled = stratified_work_budget_admission_enabled();
    let execution_shape_category_stratified =
        !stratified_enabled && execution_shape_category_stratification_needed(before);
    json!({
        "policy": if stratified_enabled {
            "private_high_transfer_stratified_work_budget_admission_v1"
        } else if execution_shape_category_stratified {
            "execution_shape_category_rotating_work_budget_admission_v1"
        } else if target_family_starvation_rescue_enabled() {
            "legacy_sequential_work_budget_admission_with_target_family_starvation_rescue_v1"
        } else {
            "legacy_sequential_work_budget_admission_v1"
        },
        "reason": if stratified_enabled {
            "preserve broad transfer pressure under Rust max-work-step trimming"
        } else if execution_shape_category_stratified {
            "rotate execution-shaped private rows by category before trimming so candidate coverage gates do not train on a distorted first-chunk sample"
        } else if target_family_starvation_rescue_enabled() {
            "preserve sequential admission while rescuing small quotas from transfer-critical families starved by the work budget"
        } else {
            "default safe policy after stratified admission improved private coverage but regressed public receiver calibration; set THESEUS_STRATIFIED_WORK_BUDGET_ADMISSION=1 for diagnostic experiments"
        },
        "stratified_enabled": stratified_enabled,
        "stratified_env_flag": "THESEUS_STRATIFIED_WORK_BUDGET_ADMISSION",
        "execution_shape_category_stratified_enabled": execution_shape_category_stratified,
        "target_family_starvation_rescue_enabled": target_family_starvation_rescue_enabled() && !stratified_enabled,
        "target_family_starvation_rescue_env_flag": "THESEUS_TARGET_FAMILY_STARVATION_RESCUE",
        "target_family_starvation_rescue_min_rows": target_family_starvation_rescue_min_rows(),
        "max_work_steps": max_work_steps,
        "estimated_work_steps_before_budget": estimated_before,
        "estimated_work_steps_after_budget": estimated_after,
        "before_family_counts": before_counts,
        "admitted_family_counts": after_counts,
        "starved_target_families": undercovered,
        "starved_target_family_count": undercovered_target_family_count,
        "undercovered_target_families": undercovered,
        "undercovered_target_family_count": undercovered_target_family_count,
        "public_tests_used": false,
        "public_solutions_used": false,
        "score_semantics": "private row admission diagnostics only; not promotion evidence"
    })
}

pub(super) fn execution_shape_category_stratification_needed(train_rows: &[CodeTask]) -> bool {
    let categories = train_rows
        .iter()
        .filter(|task| admission_family(task) == "execution_shape")
        .map(|task| task.category.as_str())
        .collect::<BTreeSet<_>>();
    categories.len() > 1
}

pub(super) fn category_stratified_work_budget_order(train_rows: &[CodeTask]) -> Vec<&CodeTask> {
    let mut grouped: BTreeMap<&str, Vec<&CodeTask>> = BTreeMap::new();
    let mut non_execution = Vec::new();
    for task in train_rows {
        if admission_family(task) == "execution_shape" {
            grouped
                .entry(task.category.as_str())
                .or_default()
                .push(task);
        } else {
            non_execution.push(task);
        }
    }
    let max_len = grouped.values().map(Vec::len).max().unwrap_or(0);
    let mut ordered = Vec::new();
    for idx in 0..max_len {
        for rows in grouped.values() {
            if let Some(task) = rows.get(idx) {
                ordered.push(*task);
            }
        }
    }
    ordered.extend(non_execution);
    ordered
}

fn rescue_starved_target_families(
    train_rows: &[CodeTask],
    mut admitted: Vec<CodeTask>,
    epochs: usize,
    train_budget: usize,
) -> Vec<CodeTask> {
    if !target_family_starvation_rescue_enabled() || train_rows.is_empty() {
        return admitted;
    }
    let rescue_targets = target_family_rescue_targets();
    let mut admitted_ids = admitted
        .iter()
        .map(|task| task.task_id.clone())
        .collect::<HashSet<_>>();
    let mut used = admitted
        .iter()
        .map(|task| task_train_step_cost(task, epochs))
        .sum::<usize>();
    let min_rows = target_family_starvation_rescue_min_rows();
    for family in rescue_targets {
        let pool_count = train_rows
            .iter()
            .filter(|task| admission_family(task) == family)
            .count();
        let admitted_count = admitted
            .iter()
            .filter(|task| admission_family(task) == family)
            .count();
        let target_rows = min_rows.min(pool_count);
        if pool_count == 0 || admitted_count >= target_rows {
            continue;
        }
        let mut candidates = train_rows
            .iter()
            .filter(|task| admission_family(task) == family)
            .collect::<Vec<_>>();
        candidates.sort_by_key(|task| (task_train_step_cost(task, epochs), task.task_id.clone()));
        for candidate in candidates.into_iter().take(target_rows.saturating_mul(2)) {
            if admitted
                .iter()
                .filter(|task| admission_family(task) == family)
                .count()
                >= target_rows
            {
                break;
            }
            if admitted_ids.contains(&candidate.task_id) {
                continue;
            }
            let cost = task_train_step_cost(candidate, epochs);
            while used.saturating_add(cost) > train_budget {
                let Some(evict_idx) = starvation_rescue_eviction_index(&admitted, epochs) else {
                    break;
                };
                let evicted = admitted.remove(evict_idx);
                used = used.saturating_sub(task_train_step_cost(&evicted, epochs));
                admitted_ids.remove(&evicted.task_id);
            }
            if used.saturating_add(cost) > train_budget {
                break;
            }
            used = used.saturating_add(cost);
            admitted_ids.insert(candidate.task_id.clone());
            admitted.push(candidate.clone());
        }
    }
    admitted
}

fn target_family_starvation_rescue_enabled() -> bool {
    std::env::var_os("THESEUS_TARGET_FAMILY_STARVATION_RESCUE")
        .map(|value| {
            let text = value.to_string_lossy().to_ascii_lowercase();
            !matches!(text.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(true)
}

fn target_family_starvation_rescue_min_rows() -> usize {
    std::env::var("THESEUS_TARGET_FAMILY_STARVATION_RESCUE_MIN_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value.clamp(1, 128))
        .unwrap_or(24)
}

fn target_family_rescue_targets() -> [&'static str; 7] {
    [
        "edge_conditions",
        "type_handling",
        "parsing",
        "return_shape_interface",
        "branch_loop_local_interface",
        "algorithm_choice",
        "recurrence_state",
    ]
}

fn starvation_rescue_eviction_index(admitted: &[CodeTask], epochs: usize) -> Option<usize> {
    admitted
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, task)| {
            matches!(
                admission_family(task),
                "execution_shape"
                    | "generated_private"
                    | "open_code"
                    | "residual_private"
                    | "high_transfer_other"
                    | "repo_repair"
                    | "other"
            )
        })
        .max_by_key(|(_, task)| task_train_step_cost(task, epochs))
        .map(|(idx, _)| idx)
}

fn stratified_work_budget_admission_enabled() -> bool {
    std::env::var_os("THESEUS_STRATIFIED_WORK_BUDGET_ADMISSION")
        .map(|value| {
            let text = value.to_string_lossy().to_ascii_lowercase();
            !matches!(text.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

fn stratified_work_budget_order(train_rows: &[CodeTask], epochs: usize) -> Vec<&CodeTask> {
    let mut buckets: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, task) in train_rows.iter().enumerate() {
        buckets
            .entry(admission_family(task).to_string())
            .or_default()
            .push(idx);
    }
    for rows in buckets.values_mut() {
        rows.sort_by_key(|idx| {
            (
                task_train_step_cost(&train_rows[*idx], epochs),
                train_rows[*idx].task_id.clone(),
            )
        });
    }
    let family_order = admission_family_order(&buckets);
    let mut positions: HashMap<String, usize> = HashMap::new();
    let mut ordered = Vec::new();
    loop {
        let mut advanced = false;
        for family in &family_order {
            let pos = *positions.get(family).unwrap_or(&0);
            if let Some(rows) = buckets.get(family) {
                if let Some(idx) = rows.get(pos) {
                    ordered.push(&train_rows[*idx]);
                    positions.insert(family.clone(), pos + 1);
                    advanced = true;
                }
            }
        }
        if !advanced {
            break;
        }
    }
    ordered
}

fn admission_family_order(buckets: &BTreeMap<String, Vec<usize>>) -> Vec<String> {
    let preferred = [
        "execution_shape",
        "algorithm_choice",
        "branch_loop_local_interface",
        "return_shape_interface",
        "edge_conditions",
        "type_handling",
        "parsing",
        "recurrence_state",
        "high_transfer_other",
        "repo_repair",
        "residual_private",
        "open_code",
        "generated_private",
        "other",
    ];
    let mut order = Vec::new();
    for family in preferred {
        if buckets.contains_key(family) {
            order.push(family.to_string());
        }
    }
    for family in buckets.keys() {
        if !order.contains(family) {
            order.push(family.clone());
        }
    }
    order
}

fn admission_family(task: &CodeTask) -> &'static str {
    let mut text = format!(
        "{} {} {} {} {} {}",
        task.category,
        task.source_id,
        task.card_id,
        task.benchmark_evidence_level,
        task.raw
            .get("high_transfer_source_jsonl")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        task.tags.join(" ")
    )
    .to_lowercase();
    if let Some(contract) = task.raw.get("decoder_contract").and_then(Value::as_object) {
        for key in [
            "type_family",
            "return_shape",
            "residual_label_hint",
            "category",
            "score_semantics",
        ] {
            if let Some(value) = contract.get(key).and_then(Value::as_str) {
                text.push(' ');
                text.push_str(&value.to_lowercase());
            }
        }
        if let Some(items) = contract
            .get("required_constructs")
            .and_then(Value::as_array)
        {
            for item in items.iter().filter_map(Value::as_str) {
                text.push(' ');
                text.push_str(&item.to_lowercase());
            }
        }
    }
    if text.contains("execution_shaped_program")
        || text.contains("execution_shaped_programs")
        || text.contains("private_exec_")
        || text.contains("file_path")
        || text.contains("subprocess")
        || text.contains("archive")
        || text.contains("csv")
        || text.contains("json")
    {
        return "execution_shape";
    }
    if text.contains("algorithmic_planning")
        || text.contains("algorithm_choice")
        || text.contains("search")
        || text.contains("selection")
        || text.contains("sort")
        || text.contains("prime")
        || text.contains("factor")
        || text.contains("median")
        || text.contains("count")
        || text.contains("dynamic")
    {
        return "algorithm_choice";
    }
    if text.contains("admissibility_and_interface")
        || text.contains("branch")
        || text.contains("loop")
        || text.contains("locals")
        || text.contains("visible_arg")
        || text.contains("full_body_required")
        || text.contains("interface")
    {
        return "branch_loop_local_interface";
    }
    if text.contains("type_and_return_shape")
        || text.contains("type_contract")
        || text.contains("return_shape")
        || text.contains("type_family")
        || text.contains("list")
        || text.contains("dict")
        || text.contains("tuple")
    {
        return "return_shape_interface";
    }
    if text.contains("edge_conditions")
        || text.contains("edge_case")
        || text.contains("empty")
        || text.contains("none")
        || text.contains("boundary")
        || text.contains("timeout")
    {
        return "edge_conditions";
    }
    if text.contains("type_handling")
        || text.contains("type_or_name_error")
        || text.contains("conversion")
        || text.contains("string_int")
    {
        return "type_handling";
    }
    if text.contains("parsing") || text.contains("parser") || text.contains("parse_") {
        return "parsing";
    }
    if text.contains("recurrence")
        || text.contains("state_drift")
        || text.contains("fib")
        || text.contains("lucas")
        || text.contains("newman_conway")
        || text.contains("bell_number")
    {
        return "recurrence_state";
    }
    if text.contains("high_transfer/private_train") {
        return "high_transfer_other";
    }
    if text.contains("repo_repair") {
        return "repo_repair";
    }
    if text.contains("residual_code_curriculum") {
        return "residual_private";
    }
    if text.contains("open_code") || text.contains("github:") {
        return "open_code";
    }
    if text.contains("local_generated_private") {
        return "generated_private";
    }
    "other"
}

fn admission_family_counts(rows: &[CodeTask]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for task in rows {
        *counts
            .entry(admission_family(task).to_string())
            .or_insert(0) += 1;
    }
    counts
}

fn task_train_step_cost(task: &CodeTask, epochs: usize) -> usize {
    let token_steps = task_token_step_cost(task);
    let repeats = state_sequence_training_repeats(task)
        .saturating_add(symliquid_training_repeats(task))
        .saturating_add(1);
    token_steps
        .saturating_mul(epochs.max(1))
        .saturating_mul(repeats.max(1))
}

fn task_token_step_cost(task: &CodeTask) -> usize {
    solution_body_tokens(task).len().saturating_add(1).max(1)
}
