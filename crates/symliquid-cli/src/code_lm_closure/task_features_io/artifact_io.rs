// Artifact IO and candidate-summary helpers for Code LM task features.
// Kept separate from feature construction so the training hot path remains readable.

use super::*;

pub(in crate::code_lm_closure) fn read_jsonl(
    path: &Path,
) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<Value>(line) {
            rows.push(value);
        }
    }
    Ok(rows)
}

pub(in crate::code_lm_closure) fn write_json(
    path: &Path,
    value: &Value,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(value)?))?;
    Ok(())
}

pub(in crate::code_lm_closure) fn write_json_compact(
    path: &Path,
    value: &Value,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer(&mut writer, value)?;
    std::io::Write::write_all(&mut writer, b"\n")?;
    Ok(())
}

pub(in crate::code_lm_closure) fn write_jsonl(
    path: &Path,
    rows: &[Value],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut text = String::new();
    for row in rows {
        text.push_str(&serde_json::to_string(row)?);
        text.push('\n');
    }
    fs::write(path, text)?;
    Ok(())
}

pub(in crate::code_lm_closure) fn append_sts_nonregression_union_candidates(
    rows: &mut Vec<Value>,
) -> usize {
    let mut existing = std::collections::HashSet::new();
    for row in rows.iter() {
        existing.insert(candidate_union_key(row));
    }

    let mut additions = Vec::new();
    for row in rows.iter() {
        if string_field(row, "phase") != "private_eval_sts_off" {
            continue;
        }
        let source_mode = string_field(row, "candidate_generation_mode");
        if source_mode.is_empty()
            || source_mode.starts_with("sts_nonregression_union_from_same_seed_non_sts::")
            || source_mode == "student_decoder_no_admissible_candidate_residual"
        {
            continue;
        }
        let mut clone = row.clone();
        let union_mode = format!("sts_nonregression_union_from_same_seed_non_sts::{source_mode}");
        if let Some(object) = clone.as_object_mut() {
            object.insert("phase".to_string(), json!("private_eval"));
            object.insert(
                "candidate_generation_mode".to_string(),
                json!(union_mode.clone()),
            );
            object.insert(
                "origin".to_string(),
                json!(format!(
                    "student_code_lm_checkpoint_v1:{union_mode}:private_eval:sts_nonregression_union"
                )),
            );
            object.insert("sts_nonregression_union_candidate".to_string(), json!(true));
            object.insert(
                "sts_nonregression_union_source_phase".to_string(),
                json!("private_eval_sts_off"),
            );
            object.insert(
                "sts_nonregression_union_source_mode".to_string(),
                json!(source_mode),
            );
            object.insert(
                "candidate_generation_contract".to_string(),
                json!("sts_nonregression_union_same_seed_non_sts_fallback_not_promotion_evidence"),
            );
            object.insert(
                "candidate_quality_accounting".to_string(),
                json!("diagnostic_sts_nonregression_union_excluded_from_promotion_quality_denominator"),
            );
            object.insert("benchmark_promotion_eligible".to_string(), json!(false));
            object.insert("loop_closure_generated".to_string(), json!(false));
            object.insert(
                "canonical_solution_seen_by_solver".to_string(),
                json!(false),
            );
            object.insert(
                "public_tests_visible_to_generator".to_string(),
                json!(false),
            );
            if let Some(provenance) = object
                .get_mut("provenance")
                .and_then(|value| value.as_object_mut())
            {
                provenance.insert("phase".to_string(), json!("private_eval"));
                provenance.insert("candidate_generation_mode".to_string(), json!(union_mode));
                provenance.insert("sts_nonregression_union_candidate".to_string(), json!(true));
                provenance.insert(
                    "sts_nonregression_union_source_phase".to_string(),
                    json!("private_eval_sts_off"),
                );
                provenance.insert("benchmark_promotion_eligible".to_string(), json!(false));
                provenance.insert(
                    "candidate_generation_contract".to_string(),
                    json!(
                        "sts_nonregression_union_same_seed_non_sts_fallback_not_promotion_evidence"
                    ),
                );
                provenance.insert("tests_used".to_string(), json!(false));
                provenance.insert("canonical_solution_used".to_string(), json!(false));
            }
        }
        let key = candidate_union_key(&clone);
        if existing.insert(key) {
            additions.push(clone);
        }
    }
    let added = additions.len();
    rows.extend(additions);
    added
}

fn candidate_union_key(row: &Value) -> (String, String, String, String) {
    (
        string_field(row, "task_id"),
        string_field(row, "phase"),
        string_field(row, "code"),
        string_field(row, "candidate_generation_mode"),
    )
}

pub(in crate::code_lm_closure) fn string_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

pub(in crate::code_lm_closure) fn count_bool(rows: &[Value], key: &str) -> usize {
    rows.iter()
        .filter(|row| row.get(key).and_then(Value::as_bool).unwrap_or(false))
        .count()
}

pub(in crate::code_lm_closure) fn candidate_generation_modes(rows: &[Value]) -> Vec<String> {
    let mut modes = Vec::new();
    for row in rows {
        let mode = string_field(row, "candidate_generation_mode");
        if !mode.is_empty() && !modes.contains(&mode) {
            modes.push(mode);
        }
    }
    modes
}

pub(in crate::code_lm_closure) fn candidate_generation_mode_counts(rows: &[Value]) -> Value {
    let mut counts = BTreeMap::new();
    for row in rows {
        let mode = string_field(row, "candidate_generation_mode");
        if mode.is_empty() {
            continue;
        }
        *counts.entry(mode).or_insert(0usize) += 1;
    }
    json!(counts)
}

pub(in crate::code_lm_closure) fn candidate_verification_cache_summary(rows: &[Value]) -> Value {
    let mut by_task: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    for row in rows {
        let task_id = string_field(row, "task_id");
        if task_id.is_empty() {
            continue;
        }
        let Some(cache) = row
            .get("candidate_verification_cache_v1")
            .or_else(|| {
                row.get("provenance")
                    .and_then(|value| value.get("candidate_verification_cache_v1"))
            })
            .and_then(Value::as_object)
        else {
            continue;
        };
        let task_metrics = by_task.entry(task_id).or_default();
        for metric in [
            "verifier_unique_body_calls",
            "verifier_cache_hits",
            "guardrail_unique_body_calls",
            "guardrail_cache_hits",
        ] {
            let value = cache.get(metric).and_then(Value::as_u64).unwrap_or(0);
            let current = task_metrics.entry(metric.to_string()).or_insert(0);
            *current = (*current).max(value);
        }
    }
    let mut totals: BTreeMap<String, u64> = BTreeMap::new();
    for task_metrics in by_task.values() {
        for (metric, value) in task_metrics {
            *totals.entry(metric.clone()).or_insert(0) += *value;
        }
    }
    let verifier_calls = *totals.get("verifier_unique_body_calls").unwrap_or(&0);
    let verifier_hits = *totals.get("verifier_cache_hits").unwrap_or(&0);
    let guardrail_calls = *totals.get("guardrail_unique_body_calls").unwrap_or(&0);
    let guardrail_hits = *totals.get("guardrail_cache_hits").unwrap_or(&0);
    json!({
        "policy": "project_theseus_candidate_verification_cache_summary_v1",
        "scope": "sum_max_per_task_from_candidate_rows",
        "task_count_with_cache": by_task.len(),
        "verifier_unique_body_calls": verifier_calls,
        "verifier_cache_hits": verifier_hits,
        "verifier_cache_hit_rate": ratio_u64(verifier_hits, verifier_hits + verifier_calls),
        "guardrail_unique_body_calls": guardrail_calls,
        "guardrail_cache_hits": guardrail_hits,
        "guardrail_cache_hit_rate": ratio_u64(guardrail_hits, guardrail_hits + guardrail_calls),
        "score_semantics": "runtime_prefilter_efficiency_only_not_capability_evidence"
    })
}

fn ratio_u64(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        ((num as f64 / den as f64) * 1_000_000.0).round() / 1_000_000.0
    }
}

pub(in crate::code_lm_closure) fn aggregate_filter_diagnostics_rejection_counts(
    rows: &[Value],
) -> Value {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for row in rows {
        let Some(rejections) = row.get("rejection_counts").and_then(Value::as_object) else {
            continue;
        };
        for (reason, count) in rejections {
            let amount = count.as_u64().unwrap_or(0) as usize;
            *counts.entry(reason.clone()).or_insert(0) += amount;
        }
    }
    json!(counts)
}

pub(in crate::code_lm_closure) fn aggregate_filter_diagnostics_mode_counts(
    rows: &[Value],
) -> Value {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for row in rows {
        let Some(modes) = row.get("raw_mode_counts").and_then(Value::as_object) else {
            continue;
        };
        for (mode, count) in modes {
            let amount = count.as_u64().unwrap_or(0) as usize;
            *counts.entry(mode.clone()).or_insert(0) += amount;
        }
    }
    json!(counts)
}

pub(in crate::code_lm_closure) fn gate(name: &str, passed: bool, evidence: Value) -> Value {
    json!({"gate": name, "passed": passed, "evidence": evidence})
}

pub(in crate::code_lm_closure) fn now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format!("unix:{secs}")
}

pub(in crate::code_lm_closure) fn rel(value: &str) -> String {
    value.replace('\\', "/")
}

pub(in crate::code_lm_closure) fn round6(value: f32) -> f32 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

pub(in crate::code_lm_closure) fn stable_hash_hex(text: &str) -> String {
    format!(
        "{:016x}{:016x}",
        stable_hash_u64(text),
        stable_hash_u64(&format!("salt:{text}"))
    )
}
