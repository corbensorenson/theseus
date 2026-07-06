// Train-once candidate fanout scheduling and shared decoder precompute policy.
// Included by candidate_fanout.rs so behavior and privacy stay unchanged.

fn parallel_shared_decoder_precompute_enabled() -> bool {
    std::env::var("THESEUS_CODE_LM_PARALLEL_SHARED_DECODER_PRECOMPUTE")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

pub(super) fn candidate_rows(
    tasks: &[CodeTask],
    expression_bank: &[ExpressionBankItem],
    body_prototypes: &BodyPrototypeModel,
    body_ngram: &BodyNgramModel,
    state_sequence_decoder: &StateSequenceDecoder,
    symliquid_state_decoder: &SymLiquidStateDecoder,
    readout: &LinearReadout,
    vocab: &Vocab,
    checkpoint_id: &str,
    seed: u64,
    candidates_per_task: usize,
    phase: &str,
    trained: bool,
    sts_streams: &StsStreamMap,
    heartbeat: Option<CandidateHeartbeat<'_>>,
    transformer_hybrid_candidates: &TransformerHybridCandidateMap,
) -> Vec<Value> {
    let mut rows = Vec::new();
    if let Some(ctx) = heartbeat.as_ref() {
        let empty_rejections = BTreeMap::new();
        write_candidate_generation_heartbeat(ctx, 0, None, 0, 0, 0, &empty_rejections, "started");
    }
    let survival_lane_only = !transformer_hybrid_candidates.is_empty()
        && transformer_hybrid_survival_lane_only_enabled();
    let legacy_trained = trained && !survival_lane_only;
    let update_every = candidate_heartbeat_update_every_tasks();
    let precompute_beams =
        precompute_batched_beam_cache_for_fanout(candidates_per_task, tasks, legacy_trained);
    let precompute_decoder_families =
        precompute_batched_decoder_family_cache_for_fanout(candidates_per_task, tasks, legacy_trained);
    let no_sts_streams = StsStreamMap::new();
    let precompute_limit = candidates_per_task.clamp(2, 8);
    let no_sts_comparator_tasks = if precompute_decoder_families {
        tasks
            .iter()
            .filter(|task| sts_streams.contains_key(&task.task_id))
            .cloned()
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let precompute_no_sts_decoder_families =
        precompute_decoder_families && !no_sts_comparator_tasks.is_empty();
    let no_sts_precompute_tasks = no_sts_comparator_tasks.as_slice();
    let parallel_precompute = parallel_shared_decoder_precompute_enabled()
        && (precompute_beams || precompute_decoder_families)
        && tasks.len() > 1;
    let shared_precompute_started = Instant::now();
    let (
        batched_beam_cache,
        batched_beam_cache_ms,
        batched_state_sequence_cache,
        batched_state_sequence_cache_ms,
        batched_symliquid_state_cache,
        batched_symliquid_state_cache_ms,
        batched_state_sequence_no_sts_cache,
        batched_state_sequence_no_sts_cache_ms,
        batched_symliquid_state_no_sts_cache,
        batched_symliquid_state_no_sts_cache_ms,
    ) = if parallel_precompute {
        std::thread::scope(|scope| {
            let beam = scope.spawn(|| {
                let started = Instant::now();
                let cache = if precompute_beams {
                    batched_beam_bodies(tasks, readout, vocab, seed, precompute_limit, sts_streams)
                } else {
                    HashMap::new()
                };
                (cache, started.elapsed().as_millis())
            });
            let state = scope.spawn(|| {
                let started = Instant::now();
                let cache = if precompute_decoder_families {
                    batched_state_sequence_bodies(
                        tasks,
                        state_sequence_decoder,
                        body_ngram,
                        vocab,
                        seed,
                        precompute_limit,
                        sts_streams,
                    )
                } else {
                    HashMap::new()
                };
                (cache, started.elapsed().as_millis())
            });
            let symliquid = scope.spawn(|| {
                let started = Instant::now();
                let cache = if precompute_decoder_families {
                    batched_symliquid_state_bodies(
                        tasks,
                        symliquid_state_decoder,
                        body_ngram,
                        vocab,
                        seed,
                        precompute_limit,
                        sts_streams,
                    )
                } else {
                    HashMap::new()
                };
                (cache, started.elapsed().as_millis())
            });
            let state_no_sts = scope.spawn(|| {
                let started = Instant::now();
                let cache = if precompute_no_sts_decoder_families {
                    batched_state_sequence_bodies(
                        no_sts_precompute_tasks,
                        state_sequence_decoder,
                        body_ngram,
                        vocab,
                        seed,
                        precompute_limit,
                        &no_sts_streams,
                    )
                } else {
                    HashMap::new()
                };
                (cache, started.elapsed().as_millis())
            });
            let symliquid_no_sts = scope.spawn(|| {
                let started = Instant::now();
                let cache = if precompute_no_sts_decoder_families {
                    batched_symliquid_state_bodies(
                        no_sts_precompute_tasks,
                        symliquid_state_decoder,
                        body_ngram,
                        vocab,
                        seed,
                        precompute_limit,
                        &no_sts_streams,
                    )
                } else {
                    HashMap::new()
                };
                (cache, started.elapsed().as_millis())
            });
            let (batched_beam_cache, batched_beam_cache_ms) =
                beam.join().unwrap_or_else(|_| (HashMap::new(), 0));
            let (batched_state_sequence_cache, batched_state_sequence_cache_ms) =
                state.join().unwrap_or_else(|_| (HashMap::new(), 0));
            let (batched_symliquid_state_cache, batched_symliquid_state_cache_ms) =
                symliquid.join().unwrap_or_else(|_| (HashMap::new(), 0));
            let (batched_state_sequence_no_sts_cache, batched_state_sequence_no_sts_cache_ms) =
                state_no_sts.join().unwrap_or_else(|_| (HashMap::new(), 0));
            let (batched_symliquid_state_no_sts_cache, batched_symliquid_state_no_sts_cache_ms) =
                symliquid_no_sts
                    .join()
                    .unwrap_or_else(|_| (HashMap::new(), 0));
            (
                batched_beam_cache,
                batched_beam_cache_ms,
                batched_state_sequence_cache,
                batched_state_sequence_cache_ms,
                batched_symliquid_state_cache,
                batched_symliquid_state_cache_ms,
                batched_state_sequence_no_sts_cache,
                batched_state_sequence_no_sts_cache_ms,
                batched_symliquid_state_no_sts_cache,
                batched_symliquid_state_no_sts_cache_ms,
            )
        })
    } else {
        let batch_started = Instant::now();
        let batched_beam_cache = if precompute_beams {
            batched_beam_bodies(tasks, readout, vocab, seed, precompute_limit, sts_streams)
        } else {
            HashMap::new()
        };
        let batched_beam_cache_ms = batch_started.elapsed().as_millis();
        let state_sequence_batch_started = Instant::now();
        let batched_state_sequence_cache = if precompute_decoder_families {
            batched_state_sequence_bodies(
                tasks,
                state_sequence_decoder,
                body_ngram,
                vocab,
                seed,
                precompute_limit,
                sts_streams,
            )
        } else {
            HashMap::new()
        };
        let batched_state_sequence_cache_ms = state_sequence_batch_started.elapsed().as_millis();
        let symliquid_state_batch_started = Instant::now();
        let batched_symliquid_state_cache = if precompute_decoder_families {
            batched_symliquid_state_bodies(
                tasks,
                symliquid_state_decoder,
                body_ngram,
                vocab,
                seed,
                precompute_limit,
                sts_streams,
            )
        } else {
            HashMap::new()
        };
        let batched_symliquid_state_cache_ms = symliquid_state_batch_started.elapsed().as_millis();
        let state_sequence_no_sts_batch_started = Instant::now();
        let batched_state_sequence_no_sts_cache = if precompute_no_sts_decoder_families {
            batched_state_sequence_bodies(
                no_sts_precompute_tasks,
                state_sequence_decoder,
                body_ngram,
                vocab,
                seed,
                precompute_limit,
                &no_sts_streams,
            )
        } else {
            HashMap::new()
        };
        let batched_state_sequence_no_sts_cache_ms =
            state_sequence_no_sts_batch_started.elapsed().as_millis();
        let symliquid_state_no_sts_batch_started = Instant::now();
        let batched_symliquid_state_no_sts_cache = if precompute_no_sts_decoder_families {
            batched_symliquid_state_bodies(
                no_sts_precompute_tasks,
                symliquid_state_decoder,
                body_ngram,
                vocab,
                seed,
                precompute_limit,
                &no_sts_streams,
            )
        } else {
            HashMap::new()
        };
        let batched_symliquid_state_no_sts_cache_ms =
            symliquid_state_no_sts_batch_started.elapsed().as_millis();
        (
            batched_beam_cache,
            batched_beam_cache_ms,
            batched_state_sequence_cache,
            batched_state_sequence_cache_ms,
            batched_symliquid_state_cache,
            batched_symliquid_state_cache_ms,
            batched_state_sequence_no_sts_cache,
            batched_state_sequence_no_sts_cache_ms,
            batched_symliquid_state_no_sts_cache,
            batched_symliquid_state_no_sts_cache_ms,
        )
    };
    let shared_decoder_precompute_wall_ms = shared_precompute_started.elapsed().as_millis();
    let worker_count = candidate_task_fanout_worker_count(tasks.len(), trained);
    let mut completed_tasks = 0usize;
    let mut emitted_rows_so_far = 0usize;
    let mut last_accepted_for_task = 0usize;
    let mut last_rejected_for_task = 0usize;
    let mut last_rejection_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut task_results = (0..tasks.len()).map(|_| None).collect::<Vec<_>>();
    let persistent_worker_pool = persistent_task_fanout_worker_pool_enabled(tasks.len(), trained);

    if persistent_worker_pool {
        let next_task = std::sync::atomic::AtomicUsize::new(0);
        let (tx, rx) = std::sync::mpsc::channel::<(usize, CandidateTaskRows)>();
        std::thread::scope(|scope| {
            for worker_id in 0..worker_count.max(1) {
                let tx = tx.clone();
                let next_task = &next_task;
                let batched_beam_cache = &batched_beam_cache;
                let batched_state_sequence_cache = &batched_state_sequence_cache;
                let batched_symliquid_state_cache = &batched_symliquid_state_cache;
                let batched_state_sequence_no_sts_cache = &batched_state_sequence_no_sts_cache;
                let batched_symliquid_state_no_sts_cache = &batched_symliquid_state_no_sts_cache;
                scope.spawn(move || loop {
                    let task_index = next_task.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if task_index >= tasks.len() {
                        break;
                    }
                    let task = &tasks[task_index];
                    let task_sts = sts_streams.get(&task.task_id);
                    let precomputed_beams = batched_beam_cache.get(&task.task_id);
                    let precomputed_state_sequence =
                        batched_state_sequence_cache.get(&task.task_id);
                    let precomputed_symliquid_state =
                        batched_symliquid_state_cache.get(&task.task_id);
                    let precomputed_state_sequence_no_sts = batched_state_sequence_no_sts_cache
                        .get(&task.task_id)
                        .or_else(|| {
                            if task_sts.is_none() {
                                precomputed_state_sequence
                            } else {
                                None
                            }
                        });
                    let precomputed_symliquid_state_no_sts = batched_symliquid_state_no_sts_cache
                        .get(&task.task_id)
                        .or_else(|| {
                            if task_sts.is_none() {
                                precomputed_symliquid_state
                            } else {
                                None
                            }
                        });
                    let task_started = Instant::now();
                    let task_result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            candidate_rows_for_task(
                                task,
                                expression_bank,
                                body_prototypes,
                                body_ngram,
                                state_sequence_decoder,
                                symliquid_state_decoder,
                                readout,
                                vocab,
                                checkpoint_id,
                                seed,
                                candidates_per_task,
                            phase,
                            trained,
                            task_sts,
                            transformer_hybrid_candidates.get(&task.task_id),
                            precomputed_beams,
                            precomputed_state_sequence,
                            precomputed_symliquid_state,
                                precomputed_state_sequence_no_sts,
                                precomputed_symliquid_state_no_sts,
                                batched_beam_cache_ms,
                                precompute_beams,
                                batched_state_sequence_cache_ms,
                                precompute_decoder_families,
                                batched_symliquid_state_cache_ms,
                                precompute_decoder_families,
                                batched_state_sequence_no_sts_cache_ms,
                                precompute_no_sts_decoder_families,
                                batched_symliquid_state_no_sts_cache_ms,
                                precompute_no_sts_decoder_families,
                                parallel_precompute,
                                shared_decoder_precompute_wall_ms,
                                task_index == 0,
                                worker_id,
                                true,
                            )
                        }))
                        .unwrap_or_else(|_| {
                            failed_candidate_task_rows(
                                task,
                                checkpoint_id,
                                phase,
                                trained,
                                task_sts.is_some(),
                                "candidate_fanout_worker_panic",
                                task_started.elapsed().as_millis(),
                                worker_id,
                                true,
                            )
                        });
                    if tx.send((task_index, task_result)).is_err() {
                        break;
                    }
                });
            }
            drop(tx);
            for _ in 0..tasks.len() {
                let Ok((task_index, task_result)) = rx.recv() else {
                    break;
                };
                let task = &tasks[task_index];
                last_accepted_for_task = task_result.accepted_count;
                last_rejected_for_task = task_result.rejected_count;
                last_rejection_counts = task_result.rejection_counts.clone();
                emitted_rows_so_far = emitted_rows_so_far.saturating_add(task_result.rows.len());
                task_results[task_index] = Some(task_result);
                completed_tasks = completed_tasks.saturating_add(1);
                if let Some(ctx) = heartbeat.as_ref() {
                    if completed_tasks == tasks.len() || completed_tasks % update_every == 0 {
                        write_candidate_generation_heartbeat(
                            ctx,
                            completed_tasks,
                            Some(task),
                            emitted_rows_so_far,
                            last_accepted_for_task,
                            last_rejected_for_task,
                            &last_rejection_counts,
                            "running",
                        );
                    }
                }
            }
        });
    } else {
        for (task_index, task) in tasks.iter().enumerate() {
            let task_sts = sts_streams.get(&task.task_id);
            let precomputed_beams = batched_beam_cache.get(&task.task_id);
            let precomputed_state_sequence = batched_state_sequence_cache.get(&task.task_id);
            let precomputed_symliquid_state = batched_symliquid_state_cache.get(&task.task_id);
            let precomputed_state_sequence_no_sts = batched_state_sequence_no_sts_cache
                .get(&task.task_id)
                .or_else(|| {
                    if task_sts.is_none() {
                        precomputed_state_sequence
                    } else {
                        None
                    }
                });
            let precomputed_symliquid_state_no_sts = batched_symliquid_state_no_sts_cache
                .get(&task.task_id)
                .or_else(|| {
                    if task_sts.is_none() {
                        precomputed_symliquid_state
                    } else {
                        None
                    }
                });
            let task_started = Instant::now();
            let task_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                candidate_rows_for_task(
                    task,
                    expression_bank,
                    body_prototypes,
                    body_ngram,
                    state_sequence_decoder,
                    symliquid_state_decoder,
                    readout,
                    vocab,
                    checkpoint_id,
                    seed,
                    candidates_per_task,
                    phase,
                    trained,
                    task_sts,
                    transformer_hybrid_candidates.get(&task.task_id),
                    precomputed_beams,
                    precomputed_state_sequence,
                    precomputed_symliquid_state,
                    precomputed_state_sequence_no_sts,
                    precomputed_symliquid_state_no_sts,
                    batched_beam_cache_ms,
                    precompute_beams,
                    batched_state_sequence_cache_ms,
                    precompute_decoder_families,
                    batched_symliquid_state_cache_ms,
                    precompute_decoder_families,
                    batched_state_sequence_no_sts_cache_ms,
                    precompute_no_sts_decoder_families,
                    batched_symliquid_state_no_sts_cache_ms,
                    precompute_no_sts_decoder_families,
                    parallel_precompute,
                    shared_decoder_precompute_wall_ms,
                    task_index == 0,
                    0,
                    false,
                )
            }))
            .unwrap_or_else(|_| {
                failed_candidate_task_rows(
                    task,
                    checkpoint_id,
                    phase,
                    trained,
                    task_sts.is_some(),
                    "candidate_fanout_task_panic",
                    task_started.elapsed().as_millis(),
                    0,
                    false,
                )
            });
            last_accepted_for_task = task_result.accepted_count;
            last_rejected_for_task = task_result.rejected_count;
            last_rejection_counts = task_result.rejection_counts.clone();
            emitted_rows_so_far = emitted_rows_so_far.saturating_add(task_result.rows.len());
            task_results[task_index] = Some(task_result);
            completed_tasks = completed_tasks.saturating_add(1);
            if let Some(ctx) = heartbeat.as_ref() {
                if completed_tasks == tasks.len() || completed_tasks % update_every == 0 {
                    write_candidate_generation_heartbeat(
                        ctx,
                        completed_tasks,
                        Some(task),
                        emitted_rows_so_far,
                        last_accepted_for_task,
                        last_rejected_for_task,
                        &last_rejection_counts,
                        "running",
                    );
                }
            }
        }
    }
    for (task_index, task_result) in task_results.into_iter().enumerate() {
        let task = &tasks[task_index];
        rows.extend(
            task_result
                .unwrap_or_else(|| {
                    failed_candidate_task_rows(
                        task,
                        checkpoint_id,
                        phase,
                        trained,
                        sts_streams.get(&task.task_id).is_some(),
                        "candidate_fanout_missing_worker_result",
                        0,
                        0,
                        persistent_worker_pool,
                    )
                })
                .rows
                .into_iter(),
        );
    }

    if let Some(ctx) = heartbeat.as_ref() {
        write_candidate_generation_heartbeat(
            ctx,
            tasks.len(),
            tasks.last(),
            rows.len(),
            last_accepted_for_task,
            last_rejected_for_task,
            &last_rejection_counts,
            "completed",
        );
    }
    rows
}

fn failed_candidate_task_rows(
    task: &CodeTask,
    checkpoint_id: &str,
    phase: &str,
    trained: bool,
    sts_conditioned: bool,
    reason: &'static str,
    elapsed_ms: u128,
    fanout_worker_id: usize,
    persistent_worker_pool_enabled: bool,
) -> CandidateTaskRows {
    let mut rejection_counts = BTreeMap::new();
    rejection_counts.insert(reason.to_string(), 1);
    let rejection_samples = vec![json!({
        "reason": reason,
        "body_preview": "candidate fanout did not complete for this task",
        "diagnostic_only": true,
    })];
    let mut task_timing_ms = BTreeMap::new();
    task_timing_ms.insert("candidate_fanout_failure_ms".to_string(), elapsed_ms);
    task_timing_ms.insert(format!("candidate_fanout_failure_{reason}"), 1);
    let mut row = no_admissible_candidate_row(
        task,
        checkpoint_id,
        phase,
        trained,
        sts_conditioned,
        &rejection_counts,
        &rejection_samples,
        &task_timing_ms,
        0,
        0,
        0,
        0,
        1,
        elapsed_ms,
        fanout_worker_id,
        persistent_worker_pool_enabled,
    );
    if let Some(object) = row.as_object_mut() {
        object.insert("candidate_fanout_failure_reason".to_string(), json!(reason));
        object.insert(
            "candidate_generation_contract".to_string(),
            json!("student_decoder_fanout_failure_residual_not_promotion_evidence"),
        );
    }
    CandidateTaskRows {
        rows: vec![row],
        accepted_count: 0,
        rejected_count: 1,
        rejection_counts,
    }
}

fn persistent_task_fanout_worker_pool_enabled(task_count: usize, trained: bool) -> bool {
    if task_count <= 1 || !trained {
        return false;
    }
    std::env::var("THESEUS_CODE_LM_PERSISTENT_TASK_FANOUT_WORKERS")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn candidate_task_fanout_worker_count(task_count: usize, trained: bool) -> usize {
    if task_count <= 1 || !trained {
        return 1;
    }
    let default_workers = std::thread::available_parallelism()
        .map(|count| count.get().saturating_sub(2).clamp(2, 8))
        .unwrap_or(4);
    let requested = std::env::var("THESEUS_CODE_LM_TASK_FANOUT_WORKERS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default_workers);
    requested.clamp(1, task_count.min(8))
}

fn precompute_batched_beam_cache_for_fanout(
    candidates_per_task: usize,
    tasks: &[CodeTask],
    trained: bool,
) -> bool {
    let task_count = tasks.len();
    if !trained || task_count == 0 {
        return false;
    }
    if let Some(enabled) = explicit_batched_beam_precompute_policy() {
        return enabled;
    }
    if skip_public_low_latency_beam_precompute(candidates_per_task, tasks) {
        return false;
    }
    if skip_private_low_latency_shared_precompute(candidates_per_task, tasks) {
        return false;
    }
    if skip_private_optional_dependency_low_latency_precompute(candidates_per_task, tasks) {
        return false;
    }
    let low_latency_single_candidate =
        low_latency_candidate_fanout_enabled() && candidates_per_task <= 1;
    if low_latency_single_candidate {
        return low_latency_beam_precompute_default_enabled(task_count);
    }
    true
}

fn explicit_batched_beam_precompute_policy() -> Option<bool> {
    std::env::var("THESEUS_CODE_LM_BATCHED_BEAM_CACHE")
        .ok()
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
}

fn precompute_batched_decoder_family_cache_for_fanout(
    candidates_per_task: usize,
    tasks: &[CodeTask],
    trained: bool,
) -> bool {
    if !trained || tasks.len() <= 1 || candidates_per_task == 0 {
        return false;
    }
    if skip_public_low_latency_beam_precompute(candidates_per_task, tasks) {
        return false;
    }
    if skip_private_low_latency_shared_precompute(candidates_per_task, tasks) {
        return false;
    }
    if skip_private_optional_dependency_low_latency_precompute(candidates_per_task, tasks) {
        return false;
    }
    if skip_private_residual_low_latency_decoder_family_precompute(candidates_per_task, tasks) {
        return false;
    }
    std::env::var("THESEUS_CODE_LM_BATCHED_DECODER_FAMILY_CACHE")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn skip_private_residual_low_latency_decoder_family_precompute(
    candidates_per_task: usize,
    tasks: &[CodeTask],
) -> bool {
    if !low_latency_candidate_fanout_enabled()
        || candidates_per_task > 8
        || tasks.is_empty()
        || tasks.iter().any(|task| task.split == "public_calibration")
        || !broad_transfer_residual_policy_enabled()
        || !tasks
            .iter()
            .all(|task| broad_transfer_residual_policy(task).active())
    {
        return false;
    }
    std::env::var("THESEUS_CODE_LM_SKIP_PRIVATE_RESIDUAL_DECODER_FAMILY_PRECOMPUTE")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn skip_private_low_latency_shared_precompute(
    candidates_per_task: usize,
    tasks: &[CodeTask],
) -> bool {
    if !low_latency_candidate_fanout_enabled()
        || candidates_per_task > private_low_latency_precompute_candidate_limit()
        || tasks.is_empty()
        || tasks.iter().any(|task| task.split == "public_calibration")
    {
        return false;
    }
    std::env::var("THESEUS_CODE_LM_SKIP_PRIVATE_LOW_LATENCY_SHARED_PRECOMPUTE")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn private_low_latency_precompute_candidate_limit() -> usize {
    std::env::var("THESEUS_CODE_LM_PRIVATE_LOW_LATENCY_PRECOMPUTE_CANDIDATE_LIMIT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .map(|value| value.clamp(1, 16))
        .unwrap_or(2)
}

fn skip_private_optional_dependency_low_latency_precompute(
    candidates_per_task: usize,
    tasks: &[CodeTask],
) -> bool {
    if !low_latency_candidate_fanout_enabled()
        || candidates_per_task > 8
        || tasks.is_empty()
        || tasks.iter().any(|task| task.split == "public_calibration")
        || !tasks.iter().all(private_optional_dependency_task)
    {
        return false;
    }
    std::env::var("THESEUS_CODE_LM_SKIP_PRIVATE_OPTIONAL_DEPENDENCY_PRECOMPUTE")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn private_optional_dependency_task(task: &CodeTask) -> bool {
    let residual_concept = task
        .raw
        .get("residual_concept")
        .and_then(Value::as_str)
        .unwrap_or("");
    task.tags.iter().any(|tag| tag == "optional_dependency")
        && (task
            .tags
            .iter()
            .any(|tag| tag == "adapter_runtime_dependency_handling")
            || residual_concept == "adapter_runtime_dependency_handling")
}

fn skip_public_low_latency_beam_precompute(candidates_per_task: usize, tasks: &[CodeTask]) -> bool {
    if !low_latency_candidate_fanout_enabled()
        || candidates_per_task > public_metadata_low_latency_candidate_limit_for_precompute()
        || !public_metadata_single_accepted_lazy_exit_for_precompute()
        || !tasks.iter().all(|task| task.split == "public_calibration")
    {
        return false;
    }
    std::env::var("THESEUS_CODE_LM_SKIP_PUBLIC_LOW_LATENCY_BEAM_PRECOMPUTE")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn public_metadata_low_latency_candidate_limit_for_precompute() -> usize {
    std::env::var("THESEUS_CODE_LM_PUBLIC_METADATA_LOW_LATENCY_CANDIDATE_LIMIT")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .map(|value| value.clamp(1, 16))
        .unwrap_or(8)
}

fn public_metadata_single_accepted_lazy_exit_for_precompute() -> bool {
    std::env::var("THESEUS_CODE_LM_PUBLIC_METADATA_SINGLE_ACCEPTED_LAZY_EXIT")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}
