// Contract-guided token decoding and STS contract bridge for Code LM candidate fanout.
// Split out so fanout staging/ranking optimizations have a narrow ownership boundary.

use super::prefilter::*;
use super::sts_bridge::*;
use super::*;

fn precomputed_contract_family_rows(
    precomputed_rows: Option<&Vec<String>>,
    budget: usize,
) -> Option<Vec<String>> {
    precomputed_rows.map(|rows| rows.iter().take(budget).cloned().collect())
}

fn record_contract_prefilter_stats(
    timing_ms: &mut BTreeMap<String, u128>,
    prefix: &str,
    stats: CheapPrefilterStats,
) {
    timing_ms.insert(
        format!("{prefix}_prefilter_input_count"),
        stats.input_count as u128,
    );
    timing_ms.insert(
        format!("{prefix}_prefilter_output_count"),
        stats.output_count as u128,
    );
    timing_ms.insert(format!("{prefix}_prefilter_budget"), stats.budget as u128);
    timing_ms.insert(
        format!("{prefix}_prefilter_cuda_ranker_used"),
        stats.used_cuda as u128,
    );
    timing_ms.insert(
        format!("{prefix}_prefilter_feature_dim"),
        stats.feature_dim as u128,
    );
}

pub(super) fn contract_guided_token_decoder_bodies(
    task: &CodeTask,
    body_ngram: &BodyNgramModel,
    state_sequence_decoder: &StateSequenceDecoder,
    symliquid_state_decoder: &SymLiquidStateDecoder,
    readout: &LinearReadout,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
    precomputed_beams: Option<&Vec<String>>,
    precomputed_state_sequence: Option<&Vec<String>>,
    precomputed_symliquid_state: Option<&Vec<String>>,
) -> Vec<String> {
    contract_guided_token_decoder_bodies_with_timing(
        task,
        body_ngram,
        state_sequence_decoder,
        symliquid_state_decoder,
        readout,
        vocab,
        seed,
        limit,
        sts_streams,
        precomputed_beams,
        precomputed_state_sequence,
        precomputed_symliquid_state,
    )
    .0
}

pub(super) fn contract_guided_token_decoder_bodies_with_timing(
    task: &CodeTask,
    body_ngram: &BodyNgramModel,
    state_sequence_decoder: &StateSequenceDecoder,
    symliquid_state_decoder: &SymLiquidStateDecoder,
    readout: &LinearReadout,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
    precomputed_beams: Option<&Vec<String>>,
    precomputed_state_sequence: Option<&Vec<String>>,
    precomputed_symliquid_state: Option<&Vec<String>>,
) -> (Vec<String>, BTreeMap<String, u128>) {
    let mut timing_ms = BTreeMap::new();
    let mut stage_started = Instant::now();
    if limit == 0 {
        return (Vec::new(), timing_ms);
    }
    let mut seen = HashSet::new();
    let mut pool = Vec::new();
    let mut verifier_cache: HashMap<String, DecoderContractVerification> = HashMap::new();
    let mut variant_cache = CandidateVariantCache::default();
    timing_ms.insert(
        "candidate_task_context_worker_id".to_string(),
        current_candidate_fanout_worker_id() as u128,
    );
    timing_ms.insert(
        "nested_branch_parallelism_suppressed".to_string(),
        nested_branch_parallelism_suppressed_for_current_task() as u128,
    );
    let low_latency_fanout = low_latency_candidate_fanout_enabled() && limit <= 8;
    let body_ngram_limit = if low_latency_fanout {
        limit.max(4)
    } else {
        limit.saturating_mul(4).max(12)
    };
    let state_limit = contract_state_sequence_family_limit(limit, low_latency_fanout);
    let symliquid_limit = contract_symliquid_state_family_limit(limit, low_latency_fanout);
    let beam_limit = if low_latency_fanout {
        limit.max(3)
    } else {
        limit.saturating_mul(2).max(8)
    };
    timing_ms.insert(
        "contract_body_ngram_family_limit".to_string(),
        body_ngram_limit as u128,
    );
    timing_ms.insert(
        "contract_state_sequence_family_limit".to_string(),
        state_limit as u128,
    );
    timing_ms.insert(
        "contract_symliquid_state_family_limit".to_string(),
        symliquid_limit as u128,
    );
    timing_ms.insert("contract_beam_family_limit".to_string(), beam_limit as u128);
    if !low_latency_fanout && contract_cheap_prepass_enabled() {
        let prepass_started = Instant::now();
        let mut cheap_pool = Vec::new();
        let mut cheap_seen = HashSet::new();
        for body in body_ngram_bodies(
            task,
            body_ngram,
            seed ^ 0xB0D1_5EED,
            body_ngram_limit.min(24),
            sts_streams,
        ) {
            if cheap_seen.insert(body.clone()) {
                cheap_pool.push(body);
            }
        }
        for body in precomputed_beams.cloned().unwrap_or_else(|| {
            beam_bodies(
                task,
                readout,
                vocab,
                seed ^ 0xBEA0_5EED,
                beam_limit,
                sts_streams,
            )
        }) {
            if cheap_seen.insert(body.clone()) {
                cheap_pool.push(body);
            }
        }
        if let Some(body) =
            greedy_or_precomputed_beam_body(task, readout, vocab, sts_streams, precomputed_beams)
        {
            if cheap_seen.insert(body.clone()) {
                cheap_pool.push(body);
            }
        }
        for body in precomputed_contract_family_rows(precomputed_symliquid_state, symliquid_limit)
            .unwrap_or_else(|| {
                symliquid_state_bodies(
                    task,
                    symliquid_state_decoder,
                    body_ngram,
                    vocab,
                    seed ^ 0x51A7_5EED,
                    symliquid_limit,
                    sts_streams,
                )
            })
        {
            if cheap_seen.insert(body.clone()) {
                cheap_pool.push(body);
            }
        }
        timing_ms.insert(
            "contract_cheap_prepass_pool_candidates".to_string(),
            cheap_pool.len() as u128,
        );
        timing_ms.insert(
            "contract_cheap_prepass_ms".to_string(),
            prepass_started.elapsed().as_millis(),
        );
        if !cheap_pool.is_empty() {
            let mut prepass_stage = Instant::now();
            let (out, prefilter_stats) = verified_low_latency_contract_outputs_cached_with_stats(
                task,
                &cheap_pool,
                sts_streams,
                limit,
                &mut verifier_cache,
            );
            record_contract_prefilter_stats(
                &mut timing_ms,
                "contract_cheap_prepass_verify",
                prefilter_stats,
            );
            record_candidate_timing(
                &mut timing_ms,
                "contract_cheap_prepass_verify",
                &mut prepass_stage,
            );
            if out.len() >= limit {
                timing_ms.insert("contract_cheap_prepass_satisfied".to_string(), 1);
                return (out, timing_ms);
            }
            timing_ms.insert("contract_cheap_prepass_satisfied".to_string(), 0);
            for body in cheap_pool {
                if seen.insert(body.clone()) {
                    pool.push(body);
                }
            }
        }
    }
    if low_latency_fanout {
        for body in body_ngram_bodies(
            task,
            body_ngram,
            seed ^ 0xB0D1_5EED,
            body_ngram_limit,
            sts_streams,
        ) {
            if seen.insert(body.clone()) {
                pool.push(body);
            }
        }
        if let Some(precomputed) = precomputed_beams {
            for body in precomputed.iter().take(beam_limit) {
                if seen.insert(body.clone()) {
                    pool.push(body.clone());
                }
            }
        }
        record_candidate_timing(
            &mut timing_ms,
            "contract_low_latency_cheap_pool_collection",
            &mut stage_started,
        );
        if !pool.is_empty() {
            let (out, prefilter_stats) = verified_low_latency_contract_outputs_cached_with_stats(
                task,
                &pool,
                sts_streams,
                limit,
                &mut verifier_cache,
            );
            record_contract_prefilter_stats(
                &mut timing_ms,
                "contract_low_latency_cheap_base_body_pass",
                prefilter_stats,
            );
            record_candidate_timing(
                &mut timing_ms,
                "contract_low_latency_cheap_base_body_pass",
                &mut stage_started,
            );
            if out.len() >= limit {
                return (out, timing_ms);
            }
        }
        let light_started = Instant::now();
        let mut light_new_candidates = 0usize;
        let beam_rows = precomputed_beams.cloned().unwrap_or_else(|| {
            beam_bodies(
                task,
                readout,
                vocab,
                seed ^ 0xBEA0_5EED,
                beam_limit,
                sts_streams,
            )
        });
        for body in beam_rows {
            if seen.insert(body.clone()) {
                pool.push(body);
                light_new_candidates += 1;
            }
        }
        if let Some(body) =
            greedy_or_precomputed_beam_body(task, readout, vocab, sts_streams, precomputed_beams)
        {
            if seen.insert(body.clone()) {
                pool.push(body);
                light_new_candidates += 1;
            }
        }
        timing_ms.insert(
            "contract_low_latency_light_family_collection".to_string(),
            light_started.elapsed().as_millis(),
        );
        timing_ms.insert(
            "contract_low_latency_light_family_new_candidates".to_string(),
            light_new_candidates as u128,
        );
        stage_started = Instant::now();
        if light_new_candidates > 0 {
            let (out, prefilter_stats) = verified_low_latency_contract_outputs_cached_with_stats(
                task,
                &pool,
                sts_streams,
                limit,
                &mut verifier_cache,
            );
            record_contract_prefilter_stats(
                &mut timing_ms,
                "contract_low_latency_light_body_pass",
                prefilter_stats,
            );
            record_candidate_timing(
                &mut timing_ms,
                "contract_low_latency_light_body_pass",
                &mut stage_started,
            );
            if out.len() >= limit {
                return (out, timing_ms);
            }
        }
    }
    let family_pools = if low_latency_fanout && parallel_contract_token_fanout_enabled() {
        std::thread::scope(|scope| {
            let state = scope.spawn(|| {
                let started = Instant::now();
                let rows =
                    precomputed_contract_family_rows(precomputed_state_sequence, state_limit)
                        .unwrap_or_else(|| {
                            state_sequence_bodies(
                                task,
                                state_sequence_decoder,
                                body_ngram,
                                vocab,
                                seed ^ 0x57A7_E5E1,
                                state_limit,
                                sts_streams,
                            )
                        });
                ("state_sequence", rows, started.elapsed().as_millis())
            });
            let symliquid = scope.spawn(|| {
                let started = Instant::now();
                let rows =
                    precomputed_contract_family_rows(precomputed_symliquid_state, symliquid_limit)
                        .unwrap_or_else(|| {
                            symliquid_state_bodies(
                                task,
                                symliquid_state_decoder,
                                body_ngram,
                                vocab,
                                seed ^ 0x51A7_5EED,
                                symliquid_limit,
                                sts_streams,
                            )
                        });
                ("symliquid_state", rows, started.elapsed().as_millis())
            });
            vec![
                state.join().unwrap_or_default(),
                symliquid.join().unwrap_or_default(),
            ]
        })
    } else if low_latency_fanout {
        vec![
            {
                let started = Instant::now();
                let rows =
                    precomputed_contract_family_rows(precomputed_state_sequence, state_limit)
                        .unwrap_or_else(|| {
                            state_sequence_bodies(
                                task,
                                state_sequence_decoder,
                                body_ngram,
                                vocab,
                                seed ^ 0x57A7_E5E1,
                                state_limit,
                                sts_streams,
                            )
                        });
                ("state_sequence", rows, started.elapsed().as_millis())
            },
            {
                let started = Instant::now();
                let rows =
                    precomputed_contract_family_rows(precomputed_symliquid_state, symliquid_limit)
                        .unwrap_or_else(|| {
                            symliquid_state_bodies(
                                task,
                                symliquid_state_decoder,
                                body_ngram,
                                vocab,
                                seed ^ 0x51A7_5EED,
                                symliquid_limit,
                                sts_streams,
                            )
                        });
                ("symliquid_state", rows, started.elapsed().as_millis())
            },
        ]
    } else if parallel_contract_token_fanout_enabled() {
        std::thread::scope(|scope| {
            let ngram = scope.spawn(|| {
                let started = Instant::now();
                let rows = body_ngram_bodies(
                    task,
                    body_ngram,
                    seed ^ 0xB0D1_5EED,
                    body_ngram_limit,
                    sts_streams,
                );
                ("ngram", rows, started.elapsed().as_millis())
            });
            let state = scope.spawn(|| {
                let started = Instant::now();
                let rows =
                    precomputed_contract_family_rows(precomputed_state_sequence, state_limit)
                        .unwrap_or_else(|| {
                            state_sequence_bodies(
                                task,
                                state_sequence_decoder,
                                body_ngram,
                                vocab,
                                seed ^ 0x57A7_E5E1,
                                state_limit,
                                sts_streams,
                            )
                        });
                ("state_sequence", rows, started.elapsed().as_millis())
            });
            let symliquid = scope.spawn(|| {
                let started = Instant::now();
                let rows =
                    precomputed_contract_family_rows(precomputed_symliquid_state, symliquid_limit)
                        .unwrap_or_else(|| {
                            symliquid_state_bodies(
                                task,
                                symliquid_state_decoder,
                                body_ngram,
                                vocab,
                                seed ^ 0x51A7_5EED,
                                symliquid_limit,
                                sts_streams,
                            )
                        });
                ("symliquid_state", rows, started.elapsed().as_millis())
            });
            let beam = scope.spawn(|| {
                let started = Instant::now();
                let rows = precomputed_beams.cloned().unwrap_or_else(|| {
                    beam_bodies(
                        task,
                        readout,
                        vocab,
                        seed ^ 0xBEA0_5EED,
                        beam_limit,
                        sts_streams,
                    )
                });
                ("beam", rows, started.elapsed().as_millis())
            });
            let greedy = scope.spawn(|| {
                let started = Instant::now();
                let rows = greedy_or_precomputed_beam_body(
                    task,
                    readout,
                    vocab,
                    sts_streams,
                    precomputed_beams,
                )
                .into_iter()
                .collect::<Vec<_>>();
                ("greedy", rows, started.elapsed().as_millis())
            });
            vec![
                ngram.join().unwrap_or_default(),
                state.join().unwrap_or_default(),
                symliquid.join().unwrap_or_default(),
                beam.join().unwrap_or_default(),
                greedy.join().unwrap_or_default(),
            ]
        })
    } else {
        vec![
            {
                let started = Instant::now();
                let rows = body_ngram_bodies(
                    task,
                    body_ngram,
                    seed ^ 0xB0D1_5EED,
                    body_ngram_limit,
                    sts_streams,
                );
                ("ngram", rows, started.elapsed().as_millis())
            },
            {
                let started = Instant::now();
                let rows =
                    precomputed_contract_family_rows(precomputed_state_sequence, state_limit)
                        .unwrap_or_else(|| {
                            state_sequence_bodies(
                                task,
                                state_sequence_decoder,
                                body_ngram,
                                vocab,
                                seed ^ 0x57A7_E5E1,
                                state_limit,
                                sts_streams,
                            )
                        });
                ("state_sequence", rows, started.elapsed().as_millis())
            },
            {
                let started = Instant::now();
                let rows =
                    precomputed_contract_family_rows(precomputed_symliquid_state, symliquid_limit)
                        .unwrap_or_else(|| {
                            symliquid_state_bodies(
                                task,
                                symliquid_state_decoder,
                                body_ngram,
                                vocab,
                                seed ^ 0x51A7_5EED,
                                symliquid_limit,
                                sts_streams,
                            )
                        });
                ("symliquid_state", rows, started.elapsed().as_millis())
            },
            {
                let started = Instant::now();
                let rows = precomputed_beams.cloned().unwrap_or_else(|| {
                    beam_bodies(
                        task,
                        readout,
                        vocab,
                        seed ^ 0xBEA0_5EED,
                        beam_limit,
                        sts_streams,
                    )
                });
                ("beam", rows, started.elapsed().as_millis())
            },
            {
                let started = Instant::now();
                let rows = greedy_or_precomputed_beam_body(
                    task,
                    readout,
                    vocab,
                    sts_streams,
                    precomputed_beams,
                )
                .into_iter()
                .collect::<Vec<_>>();
                ("greedy", rows, started.elapsed().as_millis())
            },
        ]
    };
    record_candidate_timing(
        &mut timing_ms,
        "contract_pool_collection",
        &mut stage_started,
    );
    for (family_name, family, elapsed_ms) in family_pools {
        timing_ms.insert(format!("contract_pool_family_{family_name}"), elapsed_ms);
        for body in family {
            if seen.insert(body.clone()) {
                pool.push(body);
            }
        }
    }
    if pool.is_empty() {
        return (Vec::new(), timing_ms);
    }

    if low_latency_fanout {
        let (out, prefilter_stats) = verified_low_latency_contract_outputs_cached_with_stats(
            task,
            &pool,
            sts_streams,
            limit,
            &mut verifier_cache,
        );
        record_contract_prefilter_stats(
            &mut timing_ms,
            "contract_low_latency_base_body_pass",
            prefilter_stats,
        );
        timing_ms.insert(
            "contract_low_latency_shared_verifier_cache_entries".to_string(),
            verifier_cache.len() as u128,
        );
        record_candidate_timing(
            &mut timing_ms,
            "contract_low_latency_base_body_pass",
            &mut stage_started,
        );
        if out.len() >= limit {
            return (out, timing_ms);
        }
    } else {
        record_candidate_timing(
            &mut timing_ms,
            "contract_low_latency_base_body_pass",
            &mut stage_started,
        );
    }

    let mut expanded_variants = Vec::new();
    let mut expanded_seen = HashSet::new();
    for body in pool.into_iter() {
        for variant in variant_cache.variants(task, &body) {
            if expanded_seen.insert(variant.clone()) {
                expanded_variants.push(variant);
            }
        }
    }
    let mode = if sts_streams.is_some() {
        "rust_code_lm_sts_conditioned_contract_guided_token_decoder"
    } else {
        "rust_code_lm_contract_guided_token_decoder"
    };
    let (prefiltered_variants, variant_prefilter_stats) = prefilter_bodies_before_contract_verifier(
        task,
        expanded_variants,
        sts_streams,
        limit,
        low_latency_fanout,
        mode,
    );
    record_contract_prefilter_stats(
        &mut timing_ms,
        "contract_variant_expand_and_score",
        variant_prefilter_stats,
    );
    let mut scored = Vec::new();
    for variant in prefiltered_variants {
        let verification =
            cached_decoder_contract_verification(task, &variant, sts_streams, &mut verifier_cache);
        if !token_contract_candidate_body_ok_with_verification(task, &variant, &verification) {
            continue;
        }
        let score = contract_guided_token_candidate_score_with_verification(
            task,
            &variant,
            sts_streams,
            &verification,
        );
        scored.push((score, variant));
    }
    timing_ms.insert(
        "contract_variant_cache_entries".to_string(),
        variant_cache.entries() as u128,
    );
    timing_ms.insert(
        "contract_variant_cache_hits".to_string(),
        variant_cache.hits() as u128,
    );
    timing_ms.insert(
        "contract_variant_cache_misses".to_string(),
        variant_cache.misses() as u128,
    );
    record_candidate_timing(
        &mut timing_ms,
        "contract_variant_expand_and_score",
        &mut stage_started,
    );
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.len().cmp(&b.1.len()))
            .then_with(|| a.1.cmp(&b.1))
    });

    let mut out = Vec::new();
    let mut emitted = HashSet::new();
    for (_score, body) in scored {
        if emitted.insert(body.clone()) {
            out.push(body);
        }
        if out.len() >= limit {
            break;
        }
    }
    record_candidate_timing(&mut timing_ms, "contract_sort_and_emit", &mut stage_started);
    (out, timing_ms)
}

pub(super) fn sts_conditioned_contract_token_bridge_bodies(
    task: &CodeTask,
    body_ngram: &BodyNgramModel,
    state_sequence_decoder: &StateSequenceDecoder,
    symliquid_state_decoder: &SymLiquidStateDecoder,
    readout: &LinearReadout,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: &BTreeMap<String, String>,
    precomputed_beams: Option<&Vec<String>>,
) -> Vec<String> {
    if limit == 0 {
        return Vec::new();
    }
    let mut seen = HashSet::new();
    let mut pool = Vec::new();
    let low_latency_fanout = low_latency_candidate_fanout_enabled() && limit <= 8;
    let recursive_limit = if low_latency_fanout {
        limit.max(4)
    } else {
        limit.saturating_mul(2).max(12)
    };
    let body_ngram_limit = if low_latency_fanout {
        limit.max(4)
    } else {
        limit.saturating_mul(3).max(12)
    };
    let state_limit = contract_state_sequence_family_limit(limit, low_latency_fanout);
    let symliquid_limit = contract_symliquid_state_family_limit(limit, low_latency_fanout);
    let beam_limit = if low_latency_fanout {
        limit.max(3)
    } else {
        limit.saturating_mul(2).max(8)
    };
    let family_pools = if parallel_sts_candidate_fanout_enabled() {
        std::thread::scope(|scope| {
            let contract_sts = scope.spawn(|| {
                contract_guided_token_decoder_bodies(
                    task,
                    body_ngram,
                    state_sequence_decoder,
                    symliquid_state_decoder,
                    readout,
                    vocab,
                    seed ^ 0xC0A7_57A7,
                    recursive_limit,
                    Some(sts_streams),
                    precomputed_beams,
                    None,
                    None,
                )
            });
            let contract_off = scope.spawn(|| {
                contract_guided_token_decoder_bodies(
                    task,
                    body_ngram,
                    state_sequence_decoder,
                    symliquid_state_decoder,
                    readout,
                    vocab,
                    seed ^ 0xC0A7_0FF5,
                    recursive_limit,
                    None,
                    None,
                    None,
                    None,
                )
            });
            let ngram = scope.spawn(|| {
                body_ngram_bodies(
                    task,
                    body_ngram,
                    seed ^ 0xB0D1_57A7,
                    body_ngram_limit,
                    Some(sts_streams),
                )
            });
            let state = scope.spawn(|| {
                state_sequence_bodies(
                    task,
                    state_sequence_decoder,
                    body_ngram,
                    vocab,
                    seed ^ 0x5E51_57A7,
                    state_limit,
                    Some(sts_streams),
                )
            });
            let symliquid = scope.spawn(|| {
                symliquid_state_bodies(
                    task,
                    symliquid_state_decoder,
                    body_ngram,
                    vocab,
                    seed ^ 0x51A7_C0DE,
                    symliquid_limit,
                    Some(sts_streams),
                )
            });
            let beam = scope.spawn(|| {
                beam_bodies(
                    task,
                    readout,
                    vocab,
                    seed ^ 0xBEA0_57A7,
                    beam_limit,
                    Some(sts_streams),
                )
            });
            let greedy = scope.spawn(|| {
                greedy_body(task, readout, vocab, Some(sts_streams))
                    .into_iter()
                    .collect::<Vec<_>>()
            });
            vec![
                contract_sts.join().unwrap_or_default(),
                contract_off.join().unwrap_or_default(),
                ngram.join().unwrap_or_default(),
                state.join().unwrap_or_default(),
                symliquid.join().unwrap_or_default(),
                beam.join().unwrap_or_default(),
                greedy.join().unwrap_or_default(),
            ]
        })
    } else {
        let mut family_pools = Vec::new();
        family_pools.push(contract_guided_token_decoder_bodies(
            task,
            body_ngram,
            state_sequence_decoder,
            symliquid_state_decoder,
            readout,
            vocab,
            seed ^ 0xC0A7_57A7,
            recursive_limit,
            Some(sts_streams),
            precomputed_beams,
            None,
            None,
        ));
        family_pools.push(contract_guided_token_decoder_bodies(
            task,
            body_ngram,
            state_sequence_decoder,
            symliquid_state_decoder,
            readout,
            vocab,
            seed ^ 0xC0A7_0FF5,
            recursive_limit,
            None,
            None,
            None,
            None,
        ));
        family_pools.push(body_ngram_bodies(
            task,
            body_ngram,
            seed ^ 0xB0D1_57A7,
            body_ngram_limit,
            Some(sts_streams),
        ));
        family_pools.push(state_sequence_bodies(
            task,
            state_sequence_decoder,
            body_ngram,
            vocab,
            seed ^ 0x5E51_57A7,
            state_limit,
            Some(sts_streams),
        ));
        family_pools.push(symliquid_state_bodies(
            task,
            symliquid_state_decoder,
            body_ngram,
            vocab,
            seed ^ 0x51A7_C0DE,
            symliquid_limit,
            Some(sts_streams),
        ));
        family_pools.push(beam_bodies(
            task,
            readout,
            vocab,
            seed ^ 0xBEA0_57A7,
            beam_limit,
            Some(sts_streams),
        ));
        family_pools.push(
            greedy_body(task, readout, vocab, Some(sts_streams))
                .into_iter()
                .collect::<Vec<_>>(),
        );
        family_pools
    };
    for family in family_pools {
        for body in family {
            if seen.insert(body.clone()) {
                pool.push(body);
            }
        }
    }

    sts_bridge_ranked_outputs_from_pool(task, pool, limit, sts_streams)
}

fn contract_state_sequence_family_limit(limit: usize, low_latency_fanout: bool) -> usize {
    if low_latency_fanout {
        return limit.max(4);
    }
    let cap = std::env::var("THESEUS_CODE_LM_CONTRACT_STATE_FAMILY_CAP")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(12)
        .clamp(12, 96);
    limit.saturating_mul(2).max(10).min(cap)
}

fn contract_cheap_prepass_enabled() -> bool {
    std::env::var("THESEUS_CODE_LM_CONTRACT_CHEAP_PREPASS")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(false)
}

fn contract_symliquid_state_family_limit(limit: usize, low_latency_fanout: bool) -> usize {
    if low_latency_fanout {
        return limit.max(3);
    }
    let cap = std::env::var("THESEUS_CODE_LM_CONTRACT_SYMLIQUID_FAMILY_CAP")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(24)
        .clamp(8, 96);
    limit.saturating_mul(2).max(8).min(cap)
}

pub(in crate::code_lm_closure) fn token_contract_candidate_body_ok(
    task: &CodeTask,
    body: &str,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> bool {
    let verification = decoder_contract_verifier_v1(task, body, sts_streams);
    token_contract_candidate_body_ok_with_verification(task, body, &verification)
}

pub(super) fn token_contract_candidate_body_ok_with_verification(
    task: &CodeTask,
    body: &str,
    verification: &DecoderContractVerification,
) -> bool {
    let trimmed = body.trim();
    if trimmed.is_empty()
        || trimmed.contains("raise RuntimeError")
        || natural_language_leakage_in_body(trimmed)
        || scaffold_placeholder_body(trimmed)
        || !syntax_constrained_body(trimmed)
        || !useful_generated_body_for_task(task, trimmed)
    {
        return false;
    }
    verification.passed
}

pub(in crate::code_lm_closure) fn contract_guided_token_candidate_score(
    task: &CodeTask,
    body: &str,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> f32 {
    let verifier = decoder_contract_verifier_v1(task, body, sts_streams);
    contract_guided_token_candidate_score_with_verification(task, body, sts_streams, &verifier)
}

pub(super) fn contract_guided_token_candidate_score_with_verification(
    task: &CodeTask,
    body: &str,
    sts_streams: Option<&BTreeMap<String, String>>,
    verifier: &DecoderContractVerification,
) -> f32 {
    let lowered = body.to_lowercase();
    let mut score = body_transfer_score(task, body) + beautiful_body_score(task, body);
    if verifier.passed {
        score += 6.0;
    } else {
        score -= 1.5 * verifier.reasons.len() as f32;
    }
    if visible_argument_contract_ok(task, body) {
        score += 1.4;
    }
    if return_shape_contract_ok(task, &lowered) {
        score += 1.4 + return_shape_builder_bias(task, &lowered);
    }
    if semantic_family_contract_ok(task, body) {
        score += 1.1;
    }
    let required = decoder_required_constructs(task);
    if required_construct_contract_ok_for_task(task, body, &required) {
        score += 1.2;
    }
    if body_semantically_admissible(task, body) {
        score += 1.0;
    }
    if execution_shape_contract_ok(task, body, &required) {
        score += 0.8;
    }
    score += edge_verifier_mismatch_ranker_bonus(task, body, &lowered, &required);
    if sts_streams.is_some() {
        score += sts_conditioned_rank_bias(body, sts_streams, 0.0, 1.8);
    }
    score
}

fn edge_verifier_mismatch_ranker_bonus(
    task: &CodeTask,
    body: &str,
    lowered: &str,
    required: &BTreeSet<String>,
) -> f32 {
    if !edge_verifier_mismatch_task(task, required) {
        return 0.0;
    }
    let mut bonus = 0.0;
    if edge_invalid_input_guard(lowered) {
        bonus += 0.75;
    }
    if edge_boundary_len_guard(lowered) {
        bonus += 0.55;
    }
    if edge_loop_or_comprehension(body, lowered) {
        bonus += 0.45;
    }
    if edge_local_state(lowered) {
        bonus += 0.45;
    }
    if lowered.contains("try:") && lowered.contains("except") {
        bonus += 0.25;
    }
    if edge_generic_passthrough(lowered) {
        bonus -= 1.1;
    }
    bonus
}

fn edge_verifier_mismatch_task(task: &CodeTask, required: &BTreeSet<String>) -> bool {
    if required.contains("edge_conditions")
        || required.contains("branch")
        || required.contains("locals")
        || required.contains("loop")
    {
        return true;
    }
    let hints = semantic_decoder_v2_plan_hints(task, None);
    if hints.contains("edge_conditions")
        || hints.contains("branch_loop_skeleton")
        || hints.contains("local_state")
        || hints.contains("boundary_case")
    {
        return true;
    }
    task.tags.iter().any(|tag| {
        let lowered = tag.to_lowercase();
        lowered.contains("edge_case")
            || lowered.contains("edge_contract")
            || lowered.contains("boundary")
    })
}

fn edge_invalid_input_guard(lowered: &str) -> bool {
    lowered.contains(" is none")
        || lowered.contains("not isinstance")
        || lowered.contains("if not ")
        || lowered.contains("len(")
        || lowered.contains("return []")
        || lowered.contains("return {}")
        || lowered.contains("return none")
}

fn edge_boundary_len_guard(lowered: &str) -> bool {
    lowered.contains("len(")
        || lowered.contains(" <= 0")
        || lowered.contains(" < 0")
        || lowered.contains(" >= ")
        || lowered.contains(" > ")
        || lowered.contains("empty")
}

fn edge_loop_or_comprehension(body: &str, lowered: &str) -> bool {
    body.contains("\nfor ")
        || body.contains("\nwhile ")
        || lowered.contains(" for ")
        || lowered.contains(".append(")
        || lowered.contains(".get(")
}

fn edge_local_state(lowered: &str) -> bool {
    lowered.contains("out =")
        || lowered.contains("result =")
        || lowered.contains("total =")
        || lowered.contains("current =")
        || lowered.contains("seen =")
        || lowered.contains("counts =")
        || lowered.contains("best =")
}

fn edge_generic_passthrough(lowered: &str) -> bool {
    let compact = lowered
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    compact.len() <= 2
        && compact
            .iter()
            .any(|line| matches!(*line, "return data" | "return other" | "return input"))
}
