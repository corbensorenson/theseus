use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde_json::{json, Value};

pub(super) fn candidate_task_timing_summary(candidates: &[Value]) -> Value {
    let mut seen_tasks = HashSet::new();
    let mut timing_total_ms: BTreeMap<String, u64> = BTreeMap::new();
    let mut timing_max_ms: BTreeMap<String, u64> = BTreeMap::new();
    let mut shared_timing_ms: BTreeMap<String, u64> = BTreeMap::new();
    let mut elapsed_total_ms = 0u64;
    let mut elapsed_max_ms = 0u64;
    let mut timing_task_count = 0usize;
    let mut persistent_worker_pool_task_count = 0usize;
    let mut fanout_worker_ids = BTreeSet::new();
    for (index, row) in candidates.iter().enumerate() {
        let task_key = row
            .get("task_id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("row_{index}"));
        if !seen_tasks.insert(task_key) {
            continue;
        }
        let Some(timing_row) = row
            .get("candidate_task_timing_v1")
            .and_then(Value::as_object)
        else {
            continue;
        };
        timing_task_count += 1;
        if let Some(elapsed_ms) = timing_row.get("elapsed_ms").and_then(Value::as_u64) {
            elapsed_total_ms = elapsed_total_ms.saturating_add(elapsed_ms);
            elapsed_max_ms = elapsed_max_ms.max(elapsed_ms);
        }
        if timing_row
            .get("persistent_worker_pool_enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            persistent_worker_pool_task_count = persistent_worker_pool_task_count.saturating_add(1);
        }
        if let Some(worker_id) = timing_row.get("fanout_worker_id").and_then(Value::as_u64) {
            fanout_worker_ids.insert(worker_id);
        }
        let Some(timing_ms) = timing_row.get("timing_ms").and_then(Value::as_object) else {
            continue;
        };
        for (phase, value) in timing_ms {
            if !candidate_task_runtime_timing_key(phase) {
                continue;
            }
            let Some(ms) = value.as_u64() else {
                continue;
            };
            if candidate_task_shared_timing_key(phase) {
                let current = shared_timing_ms.get(phase).copied().unwrap_or(0);
                shared_timing_ms.insert(phase.clone(), current.max(ms));
                continue;
            }
            let entry = timing_total_ms.entry(phase.clone()).or_insert(0);
            *entry = entry.saturating_add(ms);
            let current_max = timing_max_ms.get(phase).copied().unwrap_or(0);
            timing_max_ms.insert(phase.clone(), current_max.max(ms));
        }
    }
    let mut top_totals = timing_total_ms.iter().collect::<Vec<_>>();
    top_totals.sort_by(|left, right| right.1.cmp(left.1).then_with(|| left.0.cmp(right.0)));
    let top_timing_ms_total = top_totals
        .into_iter()
        .take(12)
        .map(|(phase, value)| (phase.clone(), json!(value)))
        .collect::<BTreeMap<_, _>>();
    json!({
        "task_count": timing_task_count,
        "elapsed_ms_total": elapsed_total_ms,
        "elapsed_ms_max": elapsed_max_ms,
        "persistent_worker_pool_task_count": persistent_worker_pool_task_count,
        "fanout_worker_count": fanout_worker_ids.len(),
        "fanout_worker_ids": fanout_worker_ids.into_iter().collect::<Vec<_>>(),
        "timing_ms_total": timing_total_ms,
        "timing_ms_max": timing_max_ms,
        "shared_timing_ms": shared_timing_ms,
        "top_timing_ms_total": top_timing_ms_total,
        "score_semantics": "deduplicated_per_task_runtime_profile_only_not_capability_evidence"
    })
}

pub(super) fn candidate_task_phase_category_summary(candidates: &[Value]) -> Value {
    let mut seen_tasks = HashSet::new();
    let mut categories: BTreeMap<String, u64> = BTreeMap::new();
    let mut observed_keys = BTreeMap::new();
    let mut shared_timing_ms: BTreeMap<String, u64> = BTreeMap::new();
    let mut timing_task_count = 0usize;
    for (index, row) in candidates.iter().enumerate() {
        let task_key = row
            .get("task_id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("row_{index}"));
        if !seen_tasks.insert(task_key) {
            continue;
        }
        let Some(timing_row) = row
            .get("candidate_task_timing_v1")
            .and_then(Value::as_object)
        else {
            continue;
        };
        let Some(timing_ms) = timing_row.get("timing_ms").and_then(Value::as_object) else {
            continue;
        };
        timing_task_count += 1;
        for (phase, value) in timing_ms {
            if !candidate_task_runtime_timing_key(phase) {
                continue;
            }
            let Some(ms) = value.as_u64() else {
                continue;
            };
            if ms == 0 {
                continue;
            }
            if candidate_task_shared_timing_key(phase) {
                let current = shared_timing_ms.get(phase).copied().unwrap_or(0);
                shared_timing_ms.insert(phase.clone(), current.max(ms));
                observed_keys.insert(phase.clone(), json!("shared_precompute_ms"));
                continue;
            }
            let category = candidate_task_phase_category(phase);
            *categories.entry(category.to_string()).or_insert(0) += ms;
            observed_keys.insert(phase.clone(), json!(category));
        }
    }
    let shared_precompute_wall_ms = shared_precompute_wall_ms_from_timing(
        &json!({"shared_timing_ms": shared_timing_ms.clone()}),
    );
    json!({
        "policy": "project_theseus_candidate_task_phase_categories_v1",
        "task_count": timing_task_count,
        "candidate_expansion_ms": categories.get("candidate_expansion_ms").copied().unwrap_or(0),
        "sts_conditioning_ms": categories.get("sts_conditioning_ms").copied().unwrap_or(0),
        "ranker_prefilter_ms": categories.get("ranker_prefilter_ms").copied().unwrap_or(0),
        "verifier_cache_ms": categories.get("verifier_cache_ms").copied().unwrap_or(0),
        "other_runtime_ms": categories.get("other_runtime_ms").copied().unwrap_or(0),
        "shared_precompute_ms": shared_precompute_wall_ms,
        "shared_timing_ms": shared_timing_ms,
        "observed_timing_keys": observed_keys,
        "score_semantics": "runtime_phase_category_profile_only_not_capability_evidence"
    })
}

pub(super) fn candidate_fanout_runtime_breakdown(
    phase_timing_ms: &BTreeMap<String, u128>,
    task_timing_summary: &Value,
    prefix: &str,
) -> Value {
    let candidate_generation_and_write_ms = timing_u128(
        phase_timing_ms,
        &format!("{prefix}_candidate_generation_and_write"),
    );
    let candidate_expansion_ms =
        timing_u128(phase_timing_ms, &format!("{prefix}_candidate_expansion"));
    let artifact_write_ms = timing_u128(phase_timing_ms, &format!("{prefix}_artifact_write"));
    let ranker_prefilter_verifier_summary_ms = timing_u128(
        phase_timing_ms,
        &format!("{prefix}_ranker_prefilter_verifier_cache_summary"),
    );
    let shared_precompute_wall_ms = shared_precompute_wall_ms_from_timing(task_timing_summary);
    let task_elapsed_total_ms = task_timing_summary
        .get("elapsed_ms_total")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u128;
    let task_elapsed_max_ms = task_timing_summary
        .get("elapsed_ms_max")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let task_count = task_timing_summary
        .get("task_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let fanout_worker_count = task_timing_summary
        .get("fanout_worker_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let per_task_generation_wall_ms =
        candidate_expansion_ms.saturating_sub(shared_precompute_wall_ms);
    let approximate_parallel_task_wall_ms = if fanout_worker_count > 0 {
        task_elapsed_total_ms / fanout_worker_count as u128
    } else {
        0
    };
    json!({
        "policy": "project_theseus_candidate_fanout_runtime_breakdown_v1",
        "score_semantics": "runtime_profile_only_not_capability_evidence",
        "candidate_generation_and_write_ms": candidate_generation_and_write_ms,
        "candidate_expansion_ms": candidate_expansion_ms,
        "shared_precompute_wall_ms": shared_precompute_wall_ms,
        "per_task_generation_wall_ms": per_task_generation_wall_ms,
        "artifact_write_ms": artifact_write_ms,
        "ranker_prefilter_verifier_summary_ms": ranker_prefilter_verifier_summary_ms,
        "task_elapsed_total_ms": task_elapsed_total_ms,
        "task_elapsed_max_ms": task_elapsed_max_ms,
        "task_count": task_count,
        "fanout_worker_count": fanout_worker_count,
        "approximate_parallel_task_wall_ms": approximate_parallel_task_wall_ms,
        "dominant_wall_phase": if shared_precompute_wall_ms >= per_task_generation_wall_ms {
            "shared_decoder_precompute"
        } else {
            "per_task_candidate_generation"
        },
    })
}

fn candidate_task_phase_category(phase: &str) -> &'static str {
    let phase = phase.to_ascii_lowercase();
    if phase.contains("cache") || phase.contains("verifier") {
        return "verifier_cache_ms";
    }
    if phase.contains("prefilter") || phase.contains("rank") || phase.contains("sort") {
        return "ranker_prefilter_ms";
    }
    if phase.contains("sts") || phase.contains("symliquid") {
        return "sts_conditioning_ms";
    }
    if phase.contains("candidate_expression") || phase.contains("beam") || phase.contains("ngram") {
        return "candidate_expansion_ms";
    }
    "other_runtime_ms"
}

fn candidate_task_shared_timing_key(phase: &str) -> bool {
    let phase = phase.to_ascii_lowercase();
    phase.ends_with("_shared_ms") || phase == "shared_decoder_precompute_wall_ms"
}

fn timing_u128(phase_timing_ms: &BTreeMap<String, u128>, phase: &str) -> u128 {
    phase_timing_ms.get(phase).copied().unwrap_or(0)
}

fn shared_precompute_wall_ms_from_timing(task_timing_summary: &Value) -> u128 {
    let Some(shared_timing) = task_timing_summary
        .get("shared_timing_ms")
        .and_then(Value::as_object)
    else {
        return 0;
    };
    if let Some(wall_ms) = shared_timing
        .get("shared_decoder_precompute_wall_ms")
        .and_then(Value::as_u64)
    {
        return wall_ms as u128;
    }
    shared_timing
        .values()
        .filter_map(Value::as_u64)
        .max()
        .unwrap_or(0) as u128
}

fn candidate_task_runtime_timing_key(phase: &str) -> bool {
    let phase = phase.to_ascii_lowercase();
    phase.ends_with("_ms")
        && !phase.contains("count")
        && !phase.contains("entries")
        && !phase.contains("hits")
        && !phase.contains("budget")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_precompute_is_not_summed_into_phase_categories() {
        let candidates = vec![
            json!({
                "task_id": "task_a",
                "candidate_task_timing_v1": {
                    "elapsed_ms": 25,
                    "timing_ms": {
                        "shared_decoder_precompute_wall_ms": 100,
                        "batched_beam_cache_precompute_shared_ms": 90,
                        "candidate_expression_generation_ms": 10,
                        "rank_score_and_sort_ms": 5
                    }
                }
            }),
            json!({
                "task_id": "task_b",
                "candidate_task_timing_v1": {
                    "elapsed_ms": 30,
                    "timing_ms": {
                        "shared_decoder_precompute_wall_ms": 100,
                        "batched_beam_cache_precompute_shared_ms": 90,
                        "candidate_expression_generation_ms": 12,
                        "rank_score_and_sort_ms": 6
                    }
                }
            }),
        ];

        let summary = candidate_task_phase_category_summary(&candidates);

        assert_eq!(summary["candidate_expansion_ms"].as_u64(), Some(22));
        assert_eq!(summary["ranker_prefilter_ms"].as_u64(), Some(11));
        assert_eq!(summary["verifier_cache_ms"].as_u64(), Some(0));
        assert_eq!(summary["shared_precompute_ms"].as_u64(), Some(100));
        assert_eq!(
            summary["shared_timing_ms"]["batched_beam_cache_precompute_shared_ms"].as_u64(),
            Some(90)
        );
    }

    #[test]
    fn fanout_runtime_breakdown_splits_shared_and_per_task_wall() {
        let mut phase_timing_ms = BTreeMap::new();
        phase_timing_ms.insert("private_candidate_generation_and_write".to_string(), 150);
        phase_timing_ms.insert("private_candidate_expansion".to_string(), 130);
        phase_timing_ms.insert("private_artifact_write".to_string(), 10);
        phase_timing_ms.insert(
            "private_ranker_prefilter_verifier_cache_summary".to_string(),
            5,
        );
        let task_timing_summary = json!({
            "task_count": 3,
            "elapsed_ms_total": 90,
            "elapsed_ms_max": 40,
            "fanout_worker_count": 3,
            "shared_timing_ms": {
                "shared_decoder_precompute_wall_ms": 40,
                "batched_beam_cache_precompute_shared_ms": 38
            }
        });

        let breakdown =
            candidate_fanout_runtime_breakdown(&phase_timing_ms, &task_timing_summary, "private");

        assert_eq!(
            breakdown["candidate_generation_and_write_ms"].as_u64(),
            Some(150)
        );
        assert_eq!(breakdown["shared_precompute_wall_ms"].as_u64(), Some(40));
        assert_eq!(breakdown["per_task_generation_wall_ms"].as_u64(), Some(90));
        assert_eq!(
            breakdown["approximate_parallel_task_wall_ms"].as_u64(),
            Some(30)
        );
        assert_eq!(
            breakdown["dominant_wall_phase"].as_str(),
            Some("per_task_candidate_generation")
        );
    }
}
