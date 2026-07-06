// State-sequence and parser-constrained body completion for Code LM closure.
// Kept separate from readout/token scoring so verifier-guided repair can evolve independently.

use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

mod parser_repair;
pub(super) use parser_repair::*;

const DECODER_COMPLETION_CACHE_MAX_ENTRIES: usize = 4096;
const DECODER_STATIC_FEATURE_CACHE_MAX_ENTRIES: usize = 4096;
#[allow(dead_code)]
const STATE_SEQUENCE_CUDA_READOUT_CACHE_MAX_ENTRIES: usize = 16;

static SYMLIQUID_STATE_BODY_CACHE: OnceLock<Mutex<HashMap<String, Vec<String>>>> = OnceLock::new();
static STATE_SEQUENCE_BODY_CACHE: OnceLock<Mutex<HashMap<String, Vec<String>>>> = OnceLock::new();
static STATE_SEQUENCE_STATIC_FEATURE_CACHE: OnceLock<
    Mutex<HashMap<String, Arc<Vec<(String, f32)>>>>,
> = OnceLock::new();
static SYMLIQUID_STATIC_STATE_CACHE: OnceLock<Mutex<HashMap<String, Arc<Vec<f32>>>>> =
    OnceLock::new();
#[cfg(feature = "cuda")]
static STATE_SEQUENCE_CUDA_READOUT_CACHE: OnceLock<
    Mutex<HashMap<String, Arc<StateSequenceCudaReadoutTemplate>>>,
> = OnceLock::new();
static DECODER_COMPLETION_CACHE_HITS: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_CACHE_MISSES: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_CACHE_INSERTS: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_CACHE_CLEARS: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_CUDA_SYMLIQUID_BATCHES: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_CUDA_SYMLIQUID_ROWS: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_CUDA_SYMLIQUID_MULTITASK_BATCHES: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_CUDA_SYMLIQUID_MULTITASK_ROWS: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_CUDA_STATE_SEQUENCE_BATCHES: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_CUDA_STATE_SEQUENCE_ROWS: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_CUDA_STATE_SEQUENCE_MULTITASK_BATCHES: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_CUDA_STATE_SEQUENCE_MULTITASK_ROWS: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_CUDA_STATE_SEQUENCE_FALLBACK_ROWS: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_CUDA_STATE_SEQUENCE_TEMPLATE_HITS: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_CUDA_STATE_SEQUENCE_TEMPLATE_MISSES: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_CUDA_STATE_SEQUENCE_TEMPLATE_CLEARS: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_STATE_STATIC_FEATURE_HITS: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_STATE_STATIC_FEATURE_MISSES: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_STATE_STATIC_FEATURE_INSERTS: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_STATE_STATIC_FEATURE_CLEARS: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_SYMLIQUID_STATIC_STATE_HITS: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_SYMLIQUID_STATIC_STATE_MISSES: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_SYMLIQUID_STATIC_STATE_INSERTS: AtomicUsize = AtomicUsize::new(0);
static DECODER_COMPLETION_SYMLIQUID_STATIC_STATE_CLEARS: AtomicUsize = AtomicUsize::new(0);

fn decoder_completion_cache_enabled() -> bool {
    std::env::var("THESEUS_CODE_LM_DECODER_COMPLETION_CACHE")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn decoder_static_feature_cache_enabled() -> bool {
    std::env::var("THESEUS_CODE_LM_STATIC_FEATURE_CACHE")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn cached_decoder_completion_bodies(
    cache: &OnceLock<Mutex<HashMap<String, Vec<String>>>>,
    key: String,
    compute: impl FnOnce() -> Vec<String>,
) -> Vec<String> {
    if !decoder_completion_cache_enabled() {
        return compute();
    }
    let store = cache.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = store.lock() {
        if let Some(rows) = guard.get(&key) {
            DECODER_COMPLETION_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            return rows.clone();
        }
    }
    DECODER_COMPLETION_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
    let rows = compute();
    if let Ok(mut guard) = store.lock() {
        if guard.len() >= DECODER_COMPLETION_CACHE_MAX_ENTRIES {
            guard.clear();
            DECODER_COMPLETION_CACHE_CLEARS.fetch_add(1, Ordering::Relaxed);
        }
        guard.insert(key, rows.clone());
        DECODER_COMPLETION_CACHE_INSERTS.fetch_add(1, Ordering::Relaxed);
    }
    rows
}

fn prompt_tokens_cache_hash(prompt_tokens: &BTreeSet<String>) -> u64 {
    let mut text = String::new();
    for token in prompt_tokens {
        text.push_str(token);
        text.push('\u{1f}');
    }
    stable_hash_u64(&text)
}

fn sts_stream_cache_hash(sts_streams: Option<&BTreeMap<String, String>>) -> u64 {
    let mut text = String::new();
    if let Some(streams) = sts_streams {
        for (key, value) in streams {
            text.push_str(key);
            text.push('=');
            text.push_str(value);
            text.push('\u{1e}');
        }
    }
    stable_hash_u64(&text)
}

fn task_static_feature_cache_key(
    label: &str,
    task: &CodeTask,
    prompt_tokens: &BTreeSet<String>,
    state_dim: Option<usize>,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> String {
    format!(
        "{label}:task={}:source={}:entry={}:category={}:prompt={:016x}:raw={:016x}:tokens={:016x}:state_dim={}:sts={:016x}",
        task.task_id,
        task.source_task_id,
        task.entry_point,
        task.category,
        stable_hash_u64(&task.prompt),
        stable_hash_u64(&task.raw.to_string()),
        prompt_tokens_cache_hash(prompt_tokens),
        state_dim.unwrap_or(0),
        sts_stream_cache_hash(sts_streams),
    )
}

fn cached_state_sequence_static_features(
    task: &CodeTask,
    prompt_tokens: &BTreeSet<String>,
) -> Arc<Vec<(String, f32)>> {
    if !decoder_static_feature_cache_enabled() {
        return Arc::new(state_sequence_static_features(task, prompt_tokens));
    }
    let key = task_static_feature_cache_key(
        "state_sequence_static_features",
        task,
        prompt_tokens,
        None,
        None,
    );
    let store = STATE_SEQUENCE_STATIC_FEATURE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = store.lock() {
        if let Some(features) = guard.get(&key) {
            DECODER_COMPLETION_STATE_STATIC_FEATURE_HITS.fetch_add(1, Ordering::Relaxed);
            return Arc::clone(features);
        }
    }
    DECODER_COMPLETION_STATE_STATIC_FEATURE_MISSES.fetch_add(1, Ordering::Relaxed);
    let features = Arc::new(state_sequence_static_features(task, prompt_tokens));
    if let Ok(mut guard) = store.lock() {
        if guard.len() >= DECODER_STATIC_FEATURE_CACHE_MAX_ENTRIES {
            guard.clear();
            DECODER_COMPLETION_STATE_STATIC_FEATURE_CLEARS.fetch_add(1, Ordering::Relaxed);
        }
        guard.insert(key, Arc::clone(&features));
        DECODER_COMPLETION_STATE_STATIC_FEATURE_INSERTS.fetch_add(1, Ordering::Relaxed);
    }
    features
}

fn cached_symliquid_static_state_vector(
    task: &CodeTask,
    prompt_tokens: &BTreeSet<String>,
    state_dim: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> Arc<Vec<f32>> {
    if !decoder_static_feature_cache_enabled() {
        return Arc::new(symliquid_static_state_vector_uncached(
            task,
            prompt_tokens,
            state_dim,
            sts_streams,
        ));
    }
    let key = task_static_feature_cache_key(
        "symliquid_static_state_vector",
        task,
        prompt_tokens,
        Some(state_dim),
        sts_streams,
    );
    let store = SYMLIQUID_STATIC_STATE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = store.lock() {
        if let Some(state) = guard.get(&key) {
            DECODER_COMPLETION_SYMLIQUID_STATIC_STATE_HITS.fetch_add(1, Ordering::Relaxed);
            return Arc::clone(state);
        }
    }
    DECODER_COMPLETION_SYMLIQUID_STATIC_STATE_MISSES.fetch_add(1, Ordering::Relaxed);
    let state = Arc::new(symliquid_static_state_vector_uncached(
        task,
        prompt_tokens,
        state_dim,
        sts_streams,
    ));
    if let Ok(mut guard) = store.lock() {
        if guard.len() >= DECODER_STATIC_FEATURE_CACHE_MAX_ENTRIES {
            guard.clear();
            DECODER_COMPLETION_SYMLIQUID_STATIC_STATE_CLEARS.fetch_add(1, Ordering::Relaxed);
        }
        guard.insert(key, Arc::clone(&state));
        DECODER_COMPLETION_SYMLIQUID_STATIC_STATE_INSERTS.fetch_add(1, Ordering::Relaxed);
    }
    state
}

pub(super) fn decoder_completion_cache_summary() -> Value {
    let hits = DECODER_COMPLETION_CACHE_HITS.load(Ordering::Relaxed);
    let misses = DECODER_COMPLETION_CACHE_MISSES.load(Ordering::Relaxed);
    let total = hits.saturating_add(misses);
    let hit_rate = if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    };
    serde_json::json!({
        "policy": "project_theseus_decoder_completion_cache_v1",
        "enabled": decoder_completion_cache_enabled(),
        "max_entries_per_branch_cache": DECODER_COMPLETION_CACHE_MAX_ENTRIES,
        "static_feature_cache_enabled": decoder_static_feature_cache_enabled(),
        "static_feature_cache_max_entries": DECODER_STATIC_FEATURE_CACHE_MAX_ENTRIES,
        "hit_count": hits,
        "miss_count": misses,
        "insert_count": DECODER_COMPLETION_CACHE_INSERTS.load(Ordering::Relaxed),
        "clear_count": DECODER_COMPLETION_CACHE_CLEARS.load(Ordering::Relaxed),
        "state_sequence_static_feature_cache_hit_count": DECODER_COMPLETION_STATE_STATIC_FEATURE_HITS.load(Ordering::Relaxed),
        "state_sequence_static_feature_cache_miss_count": DECODER_COMPLETION_STATE_STATIC_FEATURE_MISSES.load(Ordering::Relaxed),
        "state_sequence_static_feature_cache_insert_count": DECODER_COMPLETION_STATE_STATIC_FEATURE_INSERTS.load(Ordering::Relaxed),
        "state_sequence_static_feature_cache_clear_count": DECODER_COMPLETION_STATE_STATIC_FEATURE_CLEARS.load(Ordering::Relaxed),
        "state_sequence_static_feature_cache_entries": state_sequence_static_feature_cache_entries(),
        "symliquid_static_state_cache_hit_count": DECODER_COMPLETION_SYMLIQUID_STATIC_STATE_HITS.load(Ordering::Relaxed),
        "symliquid_static_state_cache_miss_count": DECODER_COMPLETION_SYMLIQUID_STATIC_STATE_MISSES.load(Ordering::Relaxed),
        "symliquid_static_state_cache_insert_count": DECODER_COMPLETION_SYMLIQUID_STATIC_STATE_INSERTS.load(Ordering::Relaxed),
        "symliquid_static_state_cache_clear_count": DECODER_COMPLETION_SYMLIQUID_STATIC_STATE_CLEARS.load(Ordering::Relaxed),
        "symliquid_static_state_cache_entries": symliquid_static_state_cache_entries(),
        "cuda_symliquid_state_batch_count": DECODER_COMPLETION_CUDA_SYMLIQUID_BATCHES.load(Ordering::Relaxed),
        "cuda_symliquid_state_row_count": DECODER_COMPLETION_CUDA_SYMLIQUID_ROWS.load(Ordering::Relaxed),
        "cuda_symliquid_state_multitask_batch_count": DECODER_COMPLETION_CUDA_SYMLIQUID_MULTITASK_BATCHES.load(Ordering::Relaxed),
        "cuda_symliquid_state_multitask_row_count": DECODER_COMPLETION_CUDA_SYMLIQUID_MULTITASK_ROWS.load(Ordering::Relaxed),
        "cuda_state_sequence_batch_count": DECODER_COMPLETION_CUDA_STATE_SEQUENCE_BATCHES.load(Ordering::Relaxed),
        "cuda_state_sequence_row_count": DECODER_COMPLETION_CUDA_STATE_SEQUENCE_ROWS.load(Ordering::Relaxed),
        "cuda_state_sequence_multitask_batch_count": DECODER_COMPLETION_CUDA_STATE_SEQUENCE_MULTITASK_BATCHES.load(Ordering::Relaxed),
        "cuda_state_sequence_multitask_row_count": DECODER_COMPLETION_CUDA_STATE_SEQUENCE_MULTITASK_ROWS.load(Ordering::Relaxed),
        "cuda_state_sequence_fallback_row_count": DECODER_COMPLETION_CUDA_STATE_SEQUENCE_FALLBACK_ROWS.load(Ordering::Relaxed),
        "cuda_state_sequence_readout_template_hit_count": DECODER_COMPLETION_CUDA_STATE_SEQUENCE_TEMPLATE_HITS.load(Ordering::Relaxed),
        "cuda_state_sequence_readout_template_miss_count": DECODER_COMPLETION_CUDA_STATE_SEQUENCE_TEMPLATE_MISSES.load(Ordering::Relaxed),
        "cuda_state_sequence_readout_template_clear_count": DECODER_COMPLETION_CUDA_STATE_SEQUENCE_TEMPLATE_CLEARS.load(Ordering::Relaxed),
        "cuda_state_sequence_readout_template_cache_entries": state_sequence_cuda_readout_cache_entries(),
        "cuda_readout_runtime": cuda_readout_runtime_summary(),
        "hit_rate": round6(hit_rate as f32),
        "symliquid_state_cache_entries": cache_entry_count(&SYMLIQUID_STATE_BODY_CACHE),
        "state_sequence_cache_entries": cache_entry_count(&STATE_SEQUENCE_BODY_CACHE),
        "score_semantics": "runtime_branch_reuse_only_not_capability_evidence"
    })
}

fn state_sequence_static_feature_cache_entries() -> usize {
    STATE_SEQUENCE_STATIC_FEATURE_CACHE
        .get()
        .and_then(|store| store.lock().ok().map(|guard| guard.len()))
        .unwrap_or(0)
}

fn symliquid_static_state_cache_entries() -> usize {
    SYMLIQUID_STATIC_STATE_CACHE
        .get()
        .and_then(|store| store.lock().ok().map(|guard| guard.len()))
        .unwrap_or(0)
}

fn cache_entry_count(cache: &OnceLock<Mutex<HashMap<String, Vec<String>>>>) -> usize {
    cache
        .get()
        .and_then(|store| store.lock().ok().map(|guard| guard.len()))
        .unwrap_or(0)
}

fn state_sequence_cuda_readout_cache_entries() -> usize {
    #[cfg(feature = "cuda")]
    {
        STATE_SEQUENCE_CUDA_READOUT_CACHE
            .get()
            .and_then(|store| store.lock().ok().map(|guard| guard.len()))
            .unwrap_or(0)
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

fn cuda_readout_runtime_summary() -> Value {
    #[cfg(feature = "cuda")]
    {
        let summary = symliquid_cuda::readout_cuda::linear_readout_runtime_summary_cuda();
        serde_json::json!({
            "policy": "project_theseus_cuda_readout_runtime_v1",
            "topk_session_create_count": summary.topk_session_create_count,
            "topk_session_reuse_count": summary.topk_session_reuse_count,
            "topk_session_eviction_count": summary.topk_session_eviction_count,
            "topk_session_resize_count": summary.topk_session_resize_count,
            "topk_session_prepare_count": summary.topk_session_prepare_count,
            "topk_session_max_entry_count": summary.topk_session_max_entry_count,
            "topk_thread_session_count": summary.topk_thread_session_count,
            "topk_call_count": summary.topk_call_count,
            "topk_row_count": summary.topk_row_count,
            "weighted_score_session_create_count": summary.weighted_score_session_create_count,
            "weighted_score_session_reuse_count": summary.weighted_score_session_reuse_count,
            "weighted_score_session_eviction_count": summary.weighted_score_session_eviction_count,
            "weighted_score_session_resize_count": summary.weighted_score_session_resize_count,
            "weighted_score_session_prepare_count": summary.weighted_score_session_prepare_count,
            "weighted_score_session_max_entry_count": summary.weighted_score_session_max_entry_count,
            "weighted_score_thread_session_count": summary.weighted_score_thread_session_count,
            "weighted_score_call_count": summary.weighted_score_call_count,
            "weighted_score_row_count": summary.weighted_score_row_count,
            "score_semantics": "runtime_residency_telemetry_not_capability_evidence"
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        serde_json::json!({
            "policy": "project_theseus_cuda_readout_runtime_v1",
            "enabled": false,
            "score_semantics": "runtime_residency_telemetry_not_capability_evidence"
        })
    }
}

fn decoder_completion_task_fingerprint(task: &CodeTask) -> u64 {
    stable_hash_u64(&format!(
        "{}|{}|{}|{}|{}",
        task.task_id, task.source_task_id, task.split, task.category, task.entry_point
    )) ^ stable_hash_u64(&task.prompt)
}

fn decoder_completion_sts_fingerprint(sts_streams: Option<&BTreeMap<String, String>>) -> String {
    let Some(streams) = sts_streams else {
        return "sts:none".to_string();
    };
    let rows = streams
        .iter()
        .map(|(key, value)| format!("{key}:{}", stable_hash_u64(value)))
        .collect::<Vec<_>>()
        .join("|");
    format!("sts:{}", stable_hash_u64(&rows))
}

fn symliquid_state_body_cache_key(
    task: &CodeTask,
    decoder: &SymLiquidStateDecoder,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> String {
    format!(
        "sym:v2:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
        decoder_completion_task_fingerprint(task),
        decoder.state_dim,
        decoder.update_count,
        decoder.readout.input_dim,
        decoder.readout.output_dim,
        body_ngram.counts.len(),
        vocab.id_to_token.len(),
        seed,
        limit,
        template_free_student_candidates_enabled(),
        decoder_completion_sts_fingerprint(sts_streams)
    )
}

fn state_sequence_body_cache_key(
    task: &CodeTask,
    decoder: &StateSequenceDecoder,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> String {
    format!(
        "state:v2:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
        decoder_completion_task_fingerprint(task),
        decoder.update_count,
        decoder.feature_count,
        decoder.output_dim,
        body_ngram.counts.len(),
        vocab.id_to_token.len(),
        seed,
        limit,
        template_free_student_candidates_enabled(),
        decoder_completion_sts_fingerprint(sts_streams)
    )
}

pub(super) fn symliquid_state_bodies(
    task: &CodeTask,
    decoder: &SymLiquidStateDecoder,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> Vec<String> {
    if decoder.readout.output_dim == 0 {
        return Vec::new();
    }
    let key =
        symliquid_state_body_cache_key(task, decoder, body_ngram, vocab, seed, limit, sts_streams);
    cached_decoder_completion_bodies(&SYMLIQUID_STATE_BODY_CACHE, key, || {
        symliquid_state_bodies_uncached(task, decoder, body_ngram, vocab, seed, limit, sts_streams)
    })
}

fn symliquid_state_bodies_uncached(
    task: &CodeTask,
    decoder: &SymLiquidStateDecoder,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> Vec<String> {
    if decoder.readout.output_dim == 0 {
        return Vec::new();
    }
    let prompt = prompt_tokens_with_sts(task, sts_streams);
    let prompt_alignment = prompt_alignment_lookup(&prompt);
    let static_state =
        cached_symliquid_static_state_vector(task, &prompt, decoder.state_dim, sts_streams);
    let mut beams = vec![BeamState {
        tokens: Vec::new(),
        prev2: "<BOS>".to_string(),
        prev1: "<BOS>".to_string(),
        score: 0.0,
        finished: false,
    }];
    let low_latency_fanout = low_latency_candidate_fanout_enabled() && limit <= 8;
    let beam_width = if low_latency_fanout {
        limit.clamp(1, 2)
    } else if execution_shaped_category(&task.category) {
        limit.clamp(2, 4)
    } else if task.split == "public_calibration" {
        limit.clamp(2, 5)
    } else {
        limit.clamp(2, 5)
    };
    let branch_width = if low_latency_fanout {
        vocab.id_to_token.len().min(6)
    } else if execution_shaped_category(&task.category) {
        vocab.id_to_token.len().min(8)
    } else if task.split == "public_calibration" {
        vocab.id_to_token.len().min(12)
    } else {
        vocab.id_to_token.len().min(10)
    };
    prepare_symliquid_state_cuda_readout_session(&decoder.readout, branch_width);
    let max_steps = learned_token_max_steps(
        task,
        if low_latency_fanout {
            28
        } else if task.split == "public_calibration" {
            56
        } else {
            48
        },
    );
    let token_bonus_cache =
        SymLiquidTokenBonusCache::new(task, vocab, &prompt_alignment, max_steps);
    for _step in 0..max_steps {
        let mut next = Vec::new();
        let mut active_beams = Vec::new();
        let mut feature_rows = Vec::new();
        let mut positions = Vec::new();
        for beam in &beams {
            if beam.finished {
                next.push(beam.clone());
                continue;
            }
            if let Some(options) = forced_block_token_options(&beam.tokens) {
                extend_decoder_beam_options(&mut next, task, beam, options, 6, -0.8);
                continue;
            }
            let position = learned_position_cap(&task.category, beam.tokens.len());
            feature_rows.extend(symliquid_state_vector_with_static(
                task,
                static_state.as_slice(),
                &beam.tokens,
                position,
            ));
            positions.push(position);
            active_beams.push(beam);
        }
        if active_beams.is_empty() && next.is_empty() {
            break;
        }
        if !active_beams.is_empty() {
            let Ok(features) =
                Tensor::new(active_beams.len(), decoder.readout.input_dim, feature_rows)
            else {
                break;
            };
            if let Some(token_options) = symliquid_state_cuda_token_options_batch(
                task,
                body_ngram,
                vocab,
                decoder,
                &features,
                &active_beams,
                &positions,
                &token_bonus_cache,
                branch_width,
            ) {
                for (row_idx, beam) in active_beams.into_iter().enumerate() {
                    for (token, score) in token_options
                        .get(row_idx)
                        .into_iter()
                        .flat_map(|row| row.iter().cloned())
                    {
                        extend_decoder_beam_options(
                            &mut next,
                            task,
                            beam,
                            vec![(token, score)],
                            6,
                            -0.8,
                        );
                    }
                }
            } else {
                let Ok(logits) = decoder.readout.logits(&features) else {
                    break;
                };
                for (row_idx, beam) in active_beams.into_iter().enumerate() {
                    let position = positions
                        .get(row_idx)
                        .copied()
                        .unwrap_or_else(|| learned_position_cap(&task.category, beam.tokens.len()));
                    extend_symliquid_state_beam(
                        &mut next,
                        task,
                        body_ngram,
                        vocab,
                        logits.row(row_idx),
                        beam,
                        &token_bonus_cache,
                        position,
                        branch_width,
                    );
                }
            }
        }
        if next.is_empty() {
            break;
        }
        next.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    stable_hash_u64(&format!(
                        "symliquid-state:{}:{}:{:?}",
                        seed, task.task_id, a.tokens
                    ))
                    .cmp(&stable_hash_u64(&format!(
                        "symliquid-state:{}:{}:{:?}",
                        seed, task.task_id, b.tokens
                    )))
                })
        });
        beams = next.into_iter().take(beam_width).collect();
        if beams.iter().all(|beam| beam.finished) {
            break;
        }
    }
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    beams.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for beam in beams {
        let body = join_body_tokens(&beam.tokens);
        for candidate_body in state_sequence_body_variants(task, &body) {
            if state_sequence_candidate_body_ok(task, &candidate_body)
                && seen.insert(candidate_body.clone())
            {
                out.push(candidate_body);
            }
            if out.len() >= limit {
                break;
            }
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}

pub(super) fn batched_symliquid_state_bodies(
    tasks: &[CodeTask],
    decoder: &SymLiquidStateDecoder,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: &StsStreamMap,
) -> HashMap<String, Vec<String>> {
    #[cfg(feature = "cuda")]
    {
        if tasks.len() > 1
            && limit > 0
            && decoder.readout.output_dim > 0
            && cuda_symliquid_state_decode_enabled()
        {
            if let Some(rows) = batched_symliquid_state_bodies_cuda(
                tasks,
                decoder,
                body_ngram,
                vocab,
                seed,
                limit,
                sts_streams,
            ) {
                return rows;
            }
        }
    }
    let _ = seed;
    tasks
        .iter()
        .map(|task| {
            let rows = symliquid_state_bodies(
                task,
                decoder,
                body_ngram,
                vocab,
                seed,
                limit,
                sts_streams.get(&task.task_id),
            );
            (task.task_id.clone(), rows)
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn batched_symliquid_state_bodies_cuda(
    tasks: &[CodeTask],
    decoder: &SymLiquidStateDecoder,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: &StsStreamMap,
) -> Option<HashMap<String, Vec<String>>> {
    struct BatchedSymLiquidTask<'a> {
        task: &'a CodeTask,
        static_state: Arc<Vec<f32>>,
        beams: Vec<BeamState>,
        beam_width: usize,
        branch_width: usize,
        max_steps: usize,
        token_bonus_cache: SymLiquidTokenBonusCache,
    }

    let _ = seed;
    let mut states = Vec::with_capacity(tasks.len());
    let mut max_branch_width = 1usize;
    for task in tasks {
        let task_sts = sts_streams.get(&task.task_id);
        let prompt = prompt_tokens_with_sts(task, task_sts);
        let prompt_alignment = prompt_alignment_lookup(&prompt);
        let static_state =
            cached_symliquid_static_state_vector(task, &prompt, decoder.state_dim, task_sts);
        let low_latency_fanout = low_latency_candidate_fanout_enabled() && limit <= 8;
        let beam_width = if low_latency_fanout {
            limit.clamp(1, 2)
        } else if execution_shaped_category(&task.category) {
            limit.clamp(2, 4)
        } else if task.split == "public_calibration" {
            limit.clamp(2, 5)
        } else {
            limit.clamp(2, 5)
        };
        let branch_width = if low_latency_fanout {
            vocab.id_to_token.len().min(6)
        } else if execution_shaped_category(&task.category) {
            vocab.id_to_token.len().min(8)
        } else if task.split == "public_calibration" {
            vocab.id_to_token.len().min(12)
        } else {
            vocab.id_to_token.len().min(10)
        }
        .max(1);
        max_branch_width = max_branch_width.max(branch_width);
        let max_steps = learned_token_max_steps(
            task,
            if low_latency_fanout {
                28
            } else if task.split == "public_calibration" {
                56
            } else {
                48
            },
        );
        states.push(BatchedSymLiquidTask {
            task,
            static_state,
            beams: vec![BeamState {
                tokens: Vec::new(),
                prev2: "<BOS>".to_string(),
                prev1: "<BOS>".to_string(),
                score: 0.0,
                finished: false,
            }],
            beam_width,
            branch_width,
            max_steps,
            token_bonus_cache: SymLiquidTokenBonusCache::new(
                task,
                vocab,
                &prompt_alignment,
                max_steps,
            ),
        });
    }
    if states.is_empty() {
        return Some(HashMap::new());
    }
    prepare_symliquid_state_cuda_readout_session(&decoder.readout, max_branch_width);
    let max_steps = states
        .iter()
        .map(|state| state.max_steps)
        .max()
        .unwrap_or(0);
    for step in 0..max_steps {
        let mut next_by_task: Vec<Vec<BeamState>> = vec![Vec::new(); states.len()];
        let mut active = Vec::new();
        let mut feature_rows = Vec::new();
        for (task_idx, state) in states.iter().enumerate() {
            if step >= state.max_steps {
                next_by_task[task_idx].extend(state.beams.clone());
                continue;
            }
            for beam in &state.beams {
                if beam.finished {
                    next_by_task[task_idx].push(beam.clone());
                    continue;
                }
                if let Some(options) = forced_block_token_options(&beam.tokens) {
                    extend_decoder_beam_options(
                        &mut next_by_task[task_idx],
                        state.task,
                        beam,
                        options,
                        6,
                        -0.8,
                    );
                    continue;
                }
                let position = learned_position_cap(&state.task.category, beam.tokens.len());
                feature_rows.extend(symliquid_state_vector_with_static(
                    state.task,
                    state.static_state.as_slice(),
                    &beam.tokens,
                    position,
                ));
                active.push((task_idx, beam.clone(), position));
            }
        }
        if active.is_empty() {
            if next_by_task.iter().all(Vec::is_empty) {
                break;
            }
        } else {
            let features =
                Tensor::new(active.len(), decoder.readout.input_dim, feature_rows).ok()?;
            let oversample = cuda_symliquid_state_decode_oversample();
            let topk_limit = max_branch_width.saturating_mul(oversample).clamp(
                max_branch_width.max(1),
                vocab.id_to_token.len().min(256).max(1),
            );
            let ranked_rows = symliquid_cuda::readout_cuda::linear_readout_topk_log_probs_cuda(
                &features,
                &decoder.readout,
                topk_limit,
                cuda_symliquid_state_decode_top_p(),
            )
            .ok()?;
            DECODER_COMPLETION_CUDA_SYMLIQUID_MULTITASK_BATCHES.fetch_add(1, Ordering::Relaxed);
            DECODER_COMPLETION_CUDA_SYMLIQUID_MULTITASK_ROWS
                .fetch_add(active.len(), Ordering::Relaxed);
            DECODER_COMPLETION_CUDA_SYMLIQUID_BATCHES.fetch_add(1, Ordering::Relaxed);
            DECODER_COMPLETION_CUDA_SYMLIQUID_ROWS.fetch_add(active.len(), Ordering::Relaxed);
            for (row_idx, (task_idx, beam, position)) in active.into_iter().enumerate() {
                let state = &states[task_idx];
                let Some(ranked_ids) = ranked_rows.get(row_idx) else {
                    continue;
                };
                let options = symliquid_state_token_options_from_ranked_ids(
                    state.task,
                    body_ngram,
                    vocab,
                    ranked_ids,
                    &beam.tokens,
                    &state.token_bonus_cache,
                    position,
                    state.branch_width,
                );
                extend_decoder_beam_options(
                    &mut next_by_task[task_idx],
                    state.task,
                    &beam,
                    options,
                    6,
                    -0.8,
                );
            }
        }
        let mut all_finished = true;
        for (task_idx, state) in states.iter_mut().enumerate() {
            let mut next = std::mem::take(&mut next_by_task[task_idx]);
            if next.is_empty() {
                next = state.beams.clone();
            }
            next.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        stable_hash_u64(&format!(
                            "symliquid-state-batch:{}:{}:{:?}",
                            seed, state.task.task_id, a.tokens
                        ))
                        .cmp(&stable_hash_u64(&format!(
                            "symliquid-state-batch:{}:{}:{:?}",
                            seed, state.task.task_id, b.tokens
                        )))
                    })
            });
            state.beams = next.into_iter().take(state.beam_width).collect();
            if !state.beams.iter().all(|beam| beam.finished) {
                all_finished = false;
            }
        }
        if all_finished {
            break;
        }
    }
    let mut out = HashMap::new();
    for mut state in states {
        state.beams.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut rows = Vec::new();
        let mut seen = HashSet::new();
        for beam in state.beams {
            let body = join_body_tokens(&beam.tokens);
            for candidate_body in state_sequence_body_variants(state.task, &body) {
                if state_sequence_candidate_body_ok(state.task, &candidate_body)
                    && seen.insert(candidate_body.clone())
                {
                    rows.push(candidate_body);
                }
                if rows.len() >= limit {
                    break;
                }
            }
            if rows.len() >= limit {
                break;
            }
        }
        out.insert(state.task.task_id.clone(), rows);
    }
    Some(out)
}

fn extend_symliquid_state_beam(
    next: &mut Vec<BeamState>,
    task: &CodeTask,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    logits: &[f32],
    beam: &BeamState,
    token_bonus_cache: &SymLiquidTokenBonusCache,
    position: usize,
    branch_width: usize,
) {
    let options = symliquid_state_token_options(
        task,
        body_ngram,
        vocab,
        logits,
        &beam.tokens,
        token_bonus_cache,
        position,
        branch_width,
    );
    extend_decoder_beam_options(next, task, beam, options, 6, -0.8);
}

fn extend_decoder_beam_options(
    next: &mut Vec<BeamState>,
    task: &CodeTask,
    beam: &BeamState,
    options: Vec<(String, f32)>,
    repetition_window: usize,
    repetition_penalty_value: f32,
) {
    for (token, score) in options {
        if token == "<UNK>" || !task_body_token_allowed(task, &beam.tokens, &token) {
            continue;
        }
        let mut candidate = beam.clone();
        if token == "<EOS>" {
            candidate.finished = true;
            candidate.score += score + length_bonus(candidate.tokens.len());
        } else {
            let repetition_penalty = if candidate
                .tokens
                .iter()
                .rev()
                .take(repetition_window)
                .any(|item| item == &token)
            {
                repetition_penalty_value
            } else {
                0.0
            };
            candidate.tokens.push(token.clone());
            candidate.prev2 = candidate.prev1;
            candidate.prev1 = token;
            candidate.score += score + repetition_penalty;
        }
        next.push(candidate);
    }
}

fn symliquid_state_token_options(
    task: &CodeTask,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    logits: &[f32],
    existing: &[String],
    token_bonus_cache: &SymLiquidTokenBonusCache,
    position: usize,
    limit: usize,
) -> Vec<(String, f32)> {
    let prev = existing.last().map(String::as_str).unwrap_or("<BOS>");
    if prev == ":" && body_token_allowed(existing, "<NL>") {
        return vec![("<NL>".to_string(), 50.0)];
    }
    if previous_meaningful_token(existing).as_deref() == Some(":")
        && prev == "<NL>"
        && body_token_allowed(existing, "<INDENT>")
    {
        return vec![("<INDENT>".to_string(), 50.0)];
    }
    let prev1 = existing.last().map(String::as_str).unwrap_or("<BOS>");
    let prev2 = existing
        .iter()
        .rev()
        .nth(1)
        .map(String::as_str)
        .unwrap_or("<BOS>");
    let category_guidance =
        body_ngram_category_token_scores(task, body_ngram, prev2, prev1, position);
    let mut rows = Vec::with_capacity(limit.saturating_add(1));
    for (id, logit) in logits.iter().copied().enumerate() {
        let Some(token) = vocab.id_to_token.get(id).map(String::as_str) else {
            continue;
        };
        if token == "<UNK>" || !task_body_token_allowed(task, existing, token) {
            continue;
        }
        let guidance_score = category_guidance.get(token).copied().unwrap_or_else(|| {
            if category_guidance.is_empty()
                || token == "<EOS>"
                || matches!(token, "<NL>" | "<INDENT>" | "<DEDENT>")
            {
                0.0
            } else {
                -0.85
            }
        });
        let adjusted = logit
            + guidance_score * 0.55
            + token_bonus_cache.static_bonus(id)
            + token_bonus_cache.position_bonus(task, id, token, position)
            + state_sequence_rule_bonus(task, existing, token, position);
        push_bounded_symliquid_option(&mut rows, limit, token.to_string(), adjusted, id);
    }
    rows.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
    });
    rows.into_iter()
        .map(|(token, score, _)| (token, score))
        .collect()
}

#[allow(dead_code)]
fn symliquid_state_token_options_from_ranked_ids(
    task: &CodeTask,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    ranked_ids: &[(usize, f32)],
    existing: &[String],
    token_bonus_cache: &SymLiquidTokenBonusCache,
    position: usize,
    limit: usize,
) -> Vec<(String, f32)> {
    let prev = existing.last().map(String::as_str).unwrap_or("<BOS>");
    if prev == ":" && body_token_allowed(existing, "<NL>") {
        return vec![("<NL>".to_string(), 50.0)];
    }
    if previous_meaningful_token(existing).as_deref() == Some(":")
        && prev == "<NL>"
        && body_token_allowed(existing, "<INDENT>")
    {
        return vec![("<INDENT>".to_string(), 50.0)];
    }
    let prev1 = existing.last().map(String::as_str).unwrap_or("<BOS>");
    let prev2 = existing
        .iter()
        .rev()
        .nth(1)
        .map(String::as_str)
        .unwrap_or("<BOS>");
    let category_guidance =
        body_ngram_category_token_scores(task, body_ngram, prev2, prev1, position);
    let mut rows = Vec::with_capacity(limit.saturating_add(1));
    for (id, logit) in ranked_ids.iter().copied() {
        let Some(token) = vocab.id_to_token.get(id).map(String::as_str) else {
            continue;
        };
        if token == "<UNK>" || !task_body_token_allowed(task, existing, token) {
            continue;
        }
        let guidance_score = category_guidance.get(token).copied().unwrap_or_else(|| {
            if category_guidance.is_empty()
                || token == "<EOS>"
                || matches!(token, "<NL>" | "<INDENT>" | "<DEDENT>")
            {
                0.0
            } else {
                -0.85
            }
        });
        let adjusted = logit
            + guidance_score * 0.55
            + token_bonus_cache.static_bonus(id)
            + token_bonus_cache.position_bonus(task, id, token, position)
            + state_sequence_rule_bonus(task, existing, token, position);
        push_bounded_symliquid_option(&mut rows, limit, token.to_string(), adjusted, id);
    }
    rows.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
    });
    rows.into_iter()
        .map(|(token, score, _)| (token, score))
        .collect()
}

fn symliquid_state_cuda_token_options_batch(
    task: &CodeTask,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    decoder: &SymLiquidStateDecoder,
    features: &Tensor,
    beams: &[&BeamState],
    positions: &[usize],
    token_bonus_cache: &SymLiquidTokenBonusCache,
    branch_width: usize,
) -> Option<Vec<Vec<(String, f32)>>> {
    #[cfg(feature = "cuda")]
    {
        if !cuda_symliquid_state_decode_enabled() {
            return None;
        }
        let oversample = cuda_symliquid_state_decode_oversample();
        let limit = branch_width
            .saturating_mul(oversample)
            .clamp(branch_width.max(1), vocab.id_to_token.len().min(256).max(1));
        let rows = symliquid_cuda::readout_cuda::linear_readout_topk_log_probs_cuda(
            features,
            &decoder.readout,
            limit,
            cuda_symliquid_state_decode_top_p(),
        )
        .ok()?;
        DECODER_COMPLETION_CUDA_SYMLIQUID_BATCHES.fetch_add(1, Ordering::Relaxed);
        DECODER_COMPLETION_CUDA_SYMLIQUID_ROWS.fetch_add(rows.len(), Ordering::Relaxed);
        return Some(
            rows.into_iter()
                .enumerate()
                .map(|(row_idx, row)| {
                    let beam = beams.get(row_idx).copied()?;
                    let position = positions
                        .get(row_idx)
                        .copied()
                        .unwrap_or_else(|| learned_position_cap(&task.category, beam.tokens.len()));
                    Some(symliquid_state_token_options_from_ranked_ids(
                        task,
                        body_ngram,
                        vocab,
                        &row,
                        &beam.tokens,
                        token_bonus_cache,
                        position,
                        branch_width,
                    ))
                })
                .collect::<Option<Vec<_>>>()?,
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            task,
            body_ngram,
            vocab,
            decoder,
            features,
            beams,
            positions,
            token_bonus_cache,
            branch_width,
        );
        None
    }
}

#[cfg(feature = "cuda")]
fn cuda_symliquid_state_decode_enabled() -> bool {
    std::env::var("THESEUS_CODE_LM_CUDA_STATE_SYMLIQUID")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

#[cfg(feature = "cuda")]
fn cuda_symliquid_state_decode_oversample() -> usize {
    std::env::var("THESEUS_CODE_LM_CUDA_STATE_SYMLIQUID_OVERSAMPLE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(8)
        .clamp(2, 32)
}

#[cfg(feature = "cuda")]
fn cuda_symliquid_state_decode_top_p() -> f32 {
    std::env::var("THESEUS_CODE_LM_CUDA_STATE_SYMLIQUID_TOP_P")
        .ok()
        .and_then(|value| value.trim().parse::<f32>().ok())
        .unwrap_or(1.0)
        .clamp(0.0, 1.0)
}

struct StateSequenceCudaReadoutTemplate {
    feature_index: HashMap<String, usize>,
    readout: LinearReadout,
}

pub(super) struct StateSequenceCudaScorer {
    template: Arc<StateSequenceCudaReadoutTemplate>,
    static_values: Vec<f32>,
}

impl StateSequenceCudaScorer {
    pub(super) fn input_dim(&self) -> usize {
        self.template.readout.input_dim
    }

    pub(super) fn push_feature_row(&self, dynamic_features: &[(String, f32)], out: &mut Vec<f32>) {
        let row_start = out.len();
        out.extend_from_slice(&self.static_values);
        for (key, value) in dynamic_features {
            let Some(idx) = self.template.feature_index.get(key).copied() else {
                continue;
            };
            out[row_start + idx] += *value;
        }
    }
}

pub(super) fn state_sequence_cuda_scorer(
    decoder: &StateSequenceDecoder,
    static_features: &[(String, f32)],
) -> Option<StateSequenceCudaScorer> {
    #[cfg(feature = "cuda")]
    {
        if !cuda_state_sequence_decode_enabled() || decoder.output_dim == 0 {
            return None;
        }
        let mut feature_keys = decoder.weights.keys().cloned().collect::<Vec<_>>();
        feature_keys.sort();
        let cache_key = state_sequence_cuda_readout_cache_key(decoder, &feature_keys);
        let template =
            cached_state_sequence_cuda_readout_template(decoder, &feature_keys, cache_key);
        let mut static_values = vec![0.0; template.readout.input_dim];
        for (key, value) in static_features {
            let Some(idx) = template.feature_index.get(key).copied() else {
                continue;
            };
            static_values[idx] += *value;
        }
        prepare_state_sequence_cuda_readout_session(&template.readout);
        Some(StateSequenceCudaScorer {
            template,
            static_values,
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (decoder, static_features);
        None
    }
}

#[cfg(feature = "cuda")]
fn prepare_symliquid_state_cuda_readout_session(readout: &LinearReadout, branch_width: usize) {
    if !cuda_symliquid_state_decode_enabled() {
        return;
    }
    let row_capacity = cuda_symliquid_state_decode_prepare_rows();
    let k_capacity = branch_width
        .saturating_mul(cuda_symliquid_state_decode_oversample())
        .clamp(branch_width.max(1), readout.output_dim.min(256).max(1));
    let _ = symliquid_cuda::readout_cuda::prepare_thread_readout_session_cuda(
        readout,
        row_capacity,
        k_capacity,
    );
}

#[cfg(not(feature = "cuda"))]
fn prepare_symliquid_state_cuda_readout_session(_readout: &LinearReadout, _branch_width: usize) {}

#[cfg(feature = "cuda")]
fn prepare_state_sequence_cuda_readout_session(readout: &LinearReadout) {
    if !cuda_state_sequence_decode_enabled() {
        return;
    }
    let _ = symliquid_cuda::readout_cuda::prepare_thread_readout_session_cuda(
        readout,
        cuda_state_sequence_decode_prepare_rows(),
        cuda_state_sequence_decode_prepare_k().min(readout.output_dim.max(1)),
    );
}

#[cfg(feature = "cuda")]
fn cuda_symliquid_state_decode_prepare_rows() -> usize {
    std::env::var("THESEUS_CODE_LM_CUDA_STATE_SYMLIQUID_PREPARE_ROWS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(64)
        .clamp(1, 4096)
}

#[cfg(feature = "cuda")]
fn cuda_state_sequence_decode_prepare_rows() -> usize {
    std::env::var("THESEUS_CODE_LM_CUDA_STATE_SEQUENCE_PREPARE_ROWS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(64)
        .clamp(1, 4096)
}

#[cfg(feature = "cuda")]
fn cuda_state_sequence_decode_prepare_k() -> usize {
    std::env::var("THESEUS_CODE_LM_CUDA_STATE_SEQUENCE_PREPARE_K")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(64)
        .clamp(1, 512)
}

#[cfg(feature = "cuda")]
fn cached_state_sequence_cuda_readout_template(
    decoder: &StateSequenceDecoder,
    feature_keys: &[String],
    cache_key: String,
) -> Arc<StateSequenceCudaReadoutTemplate> {
    let store = STATE_SEQUENCE_CUDA_READOUT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut guard) = store.lock() {
        if let Some(template) = guard.get(&cache_key) {
            DECODER_COMPLETION_CUDA_STATE_SEQUENCE_TEMPLATE_HITS.fetch_add(1, Ordering::Relaxed);
            return Arc::clone(template);
        }
        DECODER_COMPLETION_CUDA_STATE_SEQUENCE_TEMPLATE_MISSES.fetch_add(1, Ordering::Relaxed);
        if guard.len() >= STATE_SEQUENCE_CUDA_READOUT_CACHE_MAX_ENTRIES {
            guard.clear();
            DECODER_COMPLETION_CUDA_STATE_SEQUENCE_TEMPLATE_CLEARS.fetch_add(1, Ordering::Relaxed);
        }
        let template = Arc::new(state_sequence_cuda_readout_template(decoder, feature_keys));
        guard.insert(cache_key, Arc::clone(&template));
        return template;
    }
    DECODER_COMPLETION_CUDA_STATE_SEQUENCE_TEMPLATE_MISSES.fetch_add(1, Ordering::Relaxed);
    Arc::new(state_sequence_cuda_readout_template(decoder, feature_keys))
}

#[cfg(feature = "cuda")]
fn state_sequence_cuda_readout_template(
    decoder: &StateSequenceDecoder,
    feature_keys: &[String],
) -> StateSequenceCudaReadoutTemplate {
    let input_dim = feature_keys.len().max(1);
    let mut feature_index = HashMap::with_capacity(feature_keys.len());
    let mut readout = LinearReadout::zeros(input_dim, decoder.output_dim);
    readout.bias = decoder.bias.clone();
    readout.bias.resize(decoder.output_dim, 0.0);
    for (feature_idx, key) in feature_keys.iter().enumerate() {
        feature_index.insert(key.clone(), feature_idx);
        let Some(weights) = decoder.weights.get(key) else {
            continue;
        };
        for (out_idx, weight) in weights.iter().copied().enumerate().take(decoder.output_dim) {
            readout.weights[out_idx * input_dim + feature_idx] = weight;
        }
    }
    StateSequenceCudaReadoutTemplate {
        feature_index,
        readout,
    }
}

#[cfg(feature = "cuda")]
fn state_sequence_cuda_readout_cache_key(
    decoder: &StateSequenceDecoder,
    feature_keys: &[String],
) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    fn mix(hash: &mut u64, value: u64) {
        *hash ^= value;
        *hash = hash.wrapping_mul(0x100000001b3);
    }
    mix(&mut hash, decoder.output_dim as u64);
    mix(&mut hash, decoder.feature_count as u64);
    mix(&mut hash, decoder.update_count as u64);
    mix(&mut hash, decoder.bias.len() as u64);
    for value in &decoder.bias {
        mix(&mut hash, value.to_bits() as u64);
    }
    for key in feature_keys {
        mix(&mut hash, stable_hash_u64(key));
        if let Some(weights) = decoder.weights.get(key) {
            mix(&mut hash, weights.len() as u64);
            for value in weights.iter().take(decoder.output_dim) {
                mix(&mut hash, value.to_bits() as u64);
            }
        }
    }
    format!(
        "state-seq-cuda-readout:v1:{}:{}:{}:{}",
        decoder.output_dim, decoder.feature_count, decoder.update_count, hash
    )
}

#[allow(dead_code)]
fn state_sequence_token_options_from_ranked_ids(
    task: &CodeTask,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    ranked_ids: &[(usize, f32)],
    existing: &[String],
    prompt_tokens: &[String],
    position: usize,
    limit: usize,
) -> Vec<(String, f32)> {
    let prev = existing.last().map(String::as_str).unwrap_or("<BOS>");
    if prev == ":" && body_token_allowed(existing, "<NL>") {
        return vec![("<NL>".to_string(), 50.0)];
    }
    if previous_meaningful_token(existing).as_deref() == Some(":")
        && prev == "<NL>"
        && body_token_allowed(existing, "<INDENT>")
    {
        return vec![("<INDENT>".to_string(), 50.0)];
    }
    let prev1 = existing.last().map(String::as_str).unwrap_or("<BOS>");
    let prev2 = existing
        .iter()
        .rev()
        .nth(1)
        .map(String::as_str)
        .unwrap_or("<BOS>");
    let category_guidance =
        body_ngram_category_token_scores(task, body_ngram, prev2, prev1, position);
    let mut rows = Vec::with_capacity(limit.saturating_add(1));
    for (id, score) in ranked_ids.iter().copied() {
        let Some(token) = vocab.id_to_token.get(id).map(String::as_str) else {
            continue;
        };
        if token == "<UNK>" || !task_body_token_allowed(task, existing, token) {
            continue;
        }
        let guidance_score = category_guidance.get(token).copied().unwrap_or_else(|| {
            if category_guidance.is_empty()
                || token == "<EOS>"
                || matches!(token, "<NL>" | "<INDENT>" | "<DEDENT>")
            {
                0.0
            } else {
                -1.35
            }
        });
        let adjusted = score
            + token_alignment_bonus(token, prompt_tokens)
            + body_position_bonus(token, position)
            + category_body_token_bonus(task, token)
            + category_position_token_bonus(task, token, position)
            + state_sequence_rule_bonus(task, existing, token, position)
            + guidance_score;
        push_bounded_state_sequence_option(&mut rows, limit, token.to_string(), adjusted, id);
    }
    rows.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
    });
    rows.into_iter()
        .map(|(token, score, _)| (token, score))
        .collect()
}

pub(super) fn state_sequence_cuda_token_options_batch(
    task: &CodeTask,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    scorer: &StateSequenceCudaScorer,
    features: &Tensor,
    beams: &[&BeamState],
    prompt_tokens: &[String],
    positions: &[usize],
    branch_width: usize,
) -> Option<Vec<Vec<(String, f32)>>> {
    #[cfg(feature = "cuda")]
    {
        if !cuda_state_sequence_decode_enabled() {
            return None;
        }
        let oversample = cuda_state_sequence_decode_oversample();
        let limit = branch_width
            .saturating_mul(oversample)
            .clamp(branch_width.max(1), vocab.id_to_token.len().min(256).max(1));
        let rows = symliquid_cuda::readout_cuda::linear_readout_topk_log_probs_cuda(
            features,
            &scorer.template.readout,
            limit,
            cuda_state_sequence_decode_top_p(),
        )
        .ok()?;
        DECODER_COMPLETION_CUDA_STATE_SEQUENCE_BATCHES.fetch_add(1, Ordering::Relaxed);
        DECODER_COMPLETION_CUDA_STATE_SEQUENCE_ROWS.fetch_add(rows.len(), Ordering::Relaxed);
        return Some(
            rows.into_iter()
                .enumerate()
                .map(|(row_idx, row)| {
                    let beam = beams.get(row_idx).copied()?;
                    let position = positions
                        .get(row_idx)
                        .copied()
                        .unwrap_or_else(|| learned_position_cap(&task.category, beam.tokens.len()));
                    Some(state_sequence_token_options_from_ranked_ids(
                        task,
                        body_ngram,
                        vocab,
                        &row,
                        &beam.tokens,
                        prompt_tokens,
                        position,
                        branch_width,
                    ))
                })
                .collect::<Option<Vec<_>>>()?,
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            task,
            body_ngram,
            vocab,
            scorer,
            features,
            beams,
            prompt_tokens,
            positions,
            branch_width,
        );
        None
    }
}

#[cfg(feature = "cuda")]
fn cuda_state_sequence_decode_enabled() -> bool {
    std::env::var("THESEUS_CODE_LM_CUDA_STATE_SEQUENCE")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

#[cfg(feature = "cuda")]
fn cuda_state_sequence_decode_oversample() -> usize {
    std::env::var("THESEUS_CODE_LM_CUDA_STATE_SEQUENCE_OVERSAMPLE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(8)
        .clamp(2, 32)
}

#[cfg(feature = "cuda")]
fn cuda_state_sequence_decode_top_p() -> f32 {
    std::env::var("THESEUS_CODE_LM_CUDA_STATE_SEQUENCE_TOP_P")
        .ok()
        .and_then(|value| value.trim().parse::<f32>().ok())
        .unwrap_or(1.0)
        .clamp(0.0, 1.0)
}

fn push_bounded_symliquid_option(
    rows: &mut Vec<(String, f32, usize)>,
    limit: usize,
    token: String,
    score: f32,
    vocab_index: usize,
) {
    if limit == 0 {
        return;
    }
    if rows.len() < limit {
        rows.push((token, score, vocab_index));
        return;
    }
    let mut worst_index = 0usize;
    for idx in 1..rows.len() {
        if token_option_better(
            rows[worst_index].1,
            rows[worst_index].2,
            rows[idx].1,
            rows[idx].2,
        ) {
            worst_index = idx;
        }
    }
    if token_option_better(score, vocab_index, rows[worst_index].1, rows[worst_index].2) {
        rows[worst_index] = (token, score, vocab_index);
    }
}

fn token_option_better(
    score: f32,
    vocab_index: usize,
    other_score: f32,
    other_index: usize,
) -> bool {
    match score
        .partial_cmp(&other_score)
        .unwrap_or(std::cmp::Ordering::Less)
    {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => vocab_index < other_index,
        std::cmp::Ordering::Less => false,
    }
}

struct SymLiquidTokenBonusCache {
    static_bonus_by_id: Vec<f32>,
    position_bonus_by_position: Vec<Vec<f32>>,
}

impl SymLiquidTokenBonusCache {
    fn new(
        task: &CodeTask,
        vocab: &Vocab,
        prompt_alignment: &HashSet<String>,
        max_steps: usize,
    ) -> Self {
        let static_bonus_by_id = vocab
            .id_to_token
            .iter()
            .map(|token| {
                token_alignment_bonus_from_lookup(token, prompt_alignment)
                    + category_body_token_bonus(task, token)
            })
            .collect::<Vec<_>>();
        let max_position = learned_position_cap(&task.category, max_steps.saturating_add(1));
        let position_bonus_by_position = (0..=max_position)
            .map(|position| {
                vocab
                    .id_to_token
                    .iter()
                    .map(|token| {
                        body_position_bonus(token, position)
                            + category_position_token_bonus(task, token, position)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self {
            static_bonus_by_id,
            position_bonus_by_position,
        }
    }

    fn static_bonus(&self, id: usize) -> f32 {
        self.static_bonus_by_id.get(id).copied().unwrap_or(0.0)
    }

    fn position_bonus(&self, task: &CodeTask, id: usize, token: &str, position: usize) -> f32 {
        self.position_bonus_by_position
            .get(position)
            .and_then(|row| row.get(id))
            .copied()
            .unwrap_or_else(|| {
                body_position_bonus(token, position)
                    + category_position_token_bonus(task, token, position)
            })
    }
}

fn prompt_alignment_lookup(prompt_tokens: &BTreeSet<String>) -> HashSet<String> {
    let mut lookup = HashSet::new();
    for item in prompt_tokens {
        lookup.insert(item.clone());
        for (idx, _) in item.match_indices(':') {
            let suffix = item[idx + 1..].trim();
            if !suffix.is_empty() {
                lookup.insert(suffix.to_string());
            }
        }
    }
    lookup
}

fn token_alignment_bonus_from_lookup(token: &str, prompt_alignment: &HashSet<String>) -> f32 {
    let lowered = token.to_lowercase();
    if lowered.len() >= 2 && prompt_alignment.contains(&lowered) {
        0.25
    } else {
        0.0
    }
}

pub(super) fn symliquid_state_vector(
    task: &CodeTask,
    prompt_tokens: &BTreeSet<String>,
    existing: &[String],
    position: usize,
    state_dim: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> Vec<f32> {
    let static_state =
        cached_symliquid_static_state_vector(task, prompt_tokens, state_dim, sts_streams);
    symliquid_state_vector_with_static(task, static_state.as_slice(), existing, position)
}

fn symliquid_static_state_vector_uncached(
    task: &CodeTask,
    prompt_tokens: &BTreeSet<String>,
    state_dim: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> Vec<f32> {
    let mut state = vec![0.0; state_dim.max(64)];
    add_feature(&mut state, "bias", 1.0);
    add_feature(&mut state, &format!("category:{}", task.category), 2.0);
    add_feature(
        &mut state,
        &format!(
            "semantic_family:{}",
            category_semantic_family(&task.category)
        ),
        1.25,
    );
    if let Some(arg_count) = category_expected_arg_count(task) {
        add_feature(&mut state, &format!("expected_arg_count:{arg_count}"), 0.9);
    }
    if semantic_focus_category(&task.category) {
        add_feature(&mut state, "semantic_focus_category", 0.9);
    }
    for token in prompt_tokens.iter().take(64) {
        add_feature(&mut state, &format!("prompt:{token}"), 0.22);
        add_feature(
            &mut state,
            &format!("category_prompt:{}|{token}", task.category),
            0.18,
        );
    }
    if let Some(streams) = sts_streams {
        for (stream, weight) in [
            ("solver_stream", 0.55f32),
            ("critic_stream", 0.35f32),
            ("patch_stream", 0.75f32),
            ("residual_stream", 0.45f32),
            ("tool_stream", 0.25f32),
        ] {
            if let Some(text) = streams.get(stream) {
                add_feature(&mut state, &format!("sts_stream_present:{stream}"), weight);
                for token in tokenize_code(text).into_iter().take(48) {
                    add_feature(&mut state, &format!("sts:{stream}:{token}"), weight * 0.18);
                }
            }
        }
    }
    state
}

fn symliquid_state_vector_with_static(
    task: &CodeTask,
    static_state: &[f32],
    existing: &[String],
    position: usize,
) -> Vec<f32> {
    let mut state = static_state.to_vec();
    let capped_position = learned_position_cap(&task.category, position);
    add_feature(&mut state, &format!("pos:{capped_position}"), 0.75);
    add_feature(&mut state, &format!("pos_bucket:{}", position / 4), 0.55);
    add_feature(
        &mut state,
        &format!("indent:{}", body_indent_balance(existing).min(5)),
        0.7,
    );
    if existing.is_empty() {
        add_feature(&mut state, "slot:body_start", 1.1);
    }
    let mut prev = "<BOS>".to_string();
    for (idx, token) in existing.iter().enumerate() {
        for value in &mut state {
            *value *= 0.965;
        }
        add_feature(&mut state, &format!("emit:{token}"), 0.62);
        add_feature(&mut state, &format!("transition:{prev}->{token}"), 0.42);
        add_feature(
            &mut state,
            &format!("category_emit:{}|{token}", task.category),
            0.36,
        );
        if idx + 1 == existing.len() {
            add_feature(&mut state, &format!("prev1:{token}"), 1.4);
        }
        prev = token.clone();
    }
    if let Some(last) = existing.last() {
        add_feature(&mut state, &format!("last:{last}"), 1.1);
    }
    if let Some(prev2) = existing.iter().rev().nth(1) {
        add_feature(&mut state, &format!("prev2:{prev2}"), 0.7);
    }
    for token in current_line_tokens(existing).iter().take(12) {
        add_feature(&mut state, &format!("line_has:{token}"), 0.28);
    }
    if recurrence_category(&task.category) {
        add_feature(&mut state, "concept:recurrence_state", 1.0);
    }
    if vowel_rule_category(&task.category) {
        add_feature(&mut state, "concept:string_rule_composition", 1.0);
    }
    if digit_rotation_category(&task.category) {
        add_feature(&mut state, "concept:digit_rotation", 1.0);
    }
    let norm = state
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1.0);
    for value in &mut state {
        *value = (*value / norm).tanh();
    }
    state
}

pub(super) fn state_sequence_bodies(
    task: &CodeTask,
    decoder: &StateSequenceDecoder,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> Vec<String> {
    if decoder.output_dim == 0 {
        return Vec::new();
    }
    let key =
        state_sequence_body_cache_key(task, decoder, body_ngram, vocab, seed, limit, sts_streams);
    cached_decoder_completion_bodies(&STATE_SEQUENCE_BODY_CACHE, key, || {
        state_sequence_bodies_uncached(task, decoder, body_ngram, vocab, seed, limit, sts_streams)
    })
}

pub(super) fn batched_state_sequence_bodies(
    tasks: &[CodeTask],
    decoder: &StateSequenceDecoder,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: &StsStreamMap,
) -> HashMap<String, Vec<String>> {
    #[cfg(feature = "cuda")]
    {
        if tasks.len() > 1
            && limit > 0
            && decoder.output_dim > 0
            && cuda_state_sequence_decode_enabled()
        {
            if let Some(rows) = batched_state_sequence_bodies_cuda(
                tasks,
                decoder,
                body_ngram,
                vocab,
                seed,
                limit,
                sts_streams,
            ) {
                return rows;
            }
        }
    }
    tasks
        .iter()
        .map(|task| {
            let rows = state_sequence_bodies(
                task,
                decoder,
                body_ngram,
                vocab,
                seed,
                limit,
                sts_streams.get(&task.task_id),
            );
            (task.task_id.clone(), rows)
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn batched_state_sequence_bodies_cuda(
    tasks: &[CodeTask],
    decoder: &StateSequenceDecoder,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: &StsStreamMap,
) -> Option<HashMap<String, Vec<String>>> {
    struct BatchedStateSequenceTask<'a> {
        task: &'a CodeTask,
        prompt_tokens: Vec<String>,
        scorer: StateSequenceCudaScorer,
        beams: Vec<BeamState>,
        seeded_completed: Vec<String>,
        early_out: Option<Vec<String>>,
        beam_width: usize,
        branch_width: usize,
        max_steps: usize,
    }

    let mut states = Vec::with_capacity(tasks.len());
    let mut max_branch_width = 1usize;
    for task in tasks {
        let task_sts = sts_streams.get(&task.task_id);
        let prompt = prompt_tokens_with_sts(task, task_sts);
        let static_features = cached_state_sequence_static_features(task, &prompt);
        let scorer = state_sequence_cuda_scorer(decoder, static_features.as_slice())?;
        let prompt_tokens = prompt.iter().cloned().collect::<Vec<_>>();
        let low_latency_fanout = low_latency_candidate_fanout_enabled() && limit <= 8;
        let beam_width = if low_latency_fanout {
            limit.clamp(1, 2)
        } else if execution_shaped_category(&task.category) {
            limit.clamp(2, 4)
        } else if task.split == "public_calibration" {
            limit.clamp(2, 5)
        } else {
            limit.clamp(2, 5)
        };
        let branch_width = if low_latency_fanout {
            vocab.id_to_token.len().min(6)
        } else if execution_shaped_category(&task.category) {
            vocab.id_to_token.len().min(8)
        } else if task.split == "public_calibration" {
            vocab.id_to_token.len().min(12)
        } else {
            vocab.id_to_token.len().min(10)
        }
        .max(1);
        max_branch_width = max_branch_width.max(branch_width);
        let mut beams = vec![BeamState {
            tokens: Vec::new(),
            prev2: "<BOS>".to_string(),
            prev1: "<BOS>".to_string(),
            score: 0.0,
            finished: false,
        }];
        let seed_prefixes = if template_free_student_candidates_enabled() {
            contract_decode_seed_prefixes(task)
        } else {
            state_sequence_seed_prefixes(task)
        };
        for prefix in seed_prefixes {
            if !prefix
                .iter()
                .all(|token| vocab.token_to_id.contains_key(token))
                || !prefix_is_token_allowed(&prefix)
            {
                continue;
            }
            let prev1 = prefix
                .last()
                .cloned()
                .unwrap_or_else(|| "<BOS>".to_string());
            let prev2 = prefix
                .iter()
                .rev()
                .nth(1)
                .cloned()
                .unwrap_or_else(|| "<BOS>".to_string());
            beams.push(BeamState {
                tokens: prefix,
                prev2,
                prev1,
                score: if template_free_student_candidates_enabled() {
                    1.15
                } else {
                    0.85
                },
                finished: false,
            });
        }
        let mut seeded_completed = Vec::new();
        for beam in beams.iter().skip(1) {
            let prefix_body = join_body_tokens(&beam.tokens);
            seeded_completed.extend(state_sequence_body_variants(task, &prefix_body));
        }
        let early_out = if !seeded_completed.is_empty() {
            let mut seeded_out = Vec::new();
            let mut seen = HashSet::new();
            for candidate_body in &seeded_completed {
                if state_sequence_candidate_body_ok(task, candidate_body)
                    && seen.insert(candidate_body.clone())
                {
                    seeded_out.push(candidate_body.clone());
                }
                if seeded_out.len() >= limit {
                    break;
                }
            }
            if seeded_out.len() >= limit {
                Some(seeded_out)
            } else {
                None
            }
        } else {
            None
        };
        let max_steps = learned_token_max_steps(
            task,
            if low_latency_fanout {
                28
            } else if task.split == "public_calibration" {
                56
            } else {
                48
            },
        );
        states.push(BatchedStateSequenceTask {
            task,
            prompt_tokens,
            scorer,
            beams,
            seeded_completed,
            early_out,
            beam_width,
            branch_width,
            max_steps,
        });
    }
    if states.is_empty() {
        return Some(HashMap::new());
    }
    let readout = states.first()?.scorer.template.readout.clone();
    prepare_state_sequence_cuda_readout_session(&readout);
    let max_steps = states
        .iter()
        .filter(|state| state.early_out.is_none())
        .map(|state| state.max_steps)
        .max()
        .unwrap_or(0);
    for _step in 0..max_steps {
        let mut next_by_task: Vec<Vec<BeamState>> = vec![Vec::new(); states.len()];
        let mut active = Vec::new();
        let mut feature_rows = Vec::new();
        for (task_idx, state) in states.iter().enumerate() {
            if state.early_out.is_some() {
                continue;
            }
            for beam in &state.beams {
                if beam.finished {
                    next_by_task[task_idx].push(beam.clone());
                    continue;
                }
                let effective_position =
                    learned_position_cap(&state.task.category, beam.tokens.len());
                if let Some(options) = forced_block_token_options(&beam.tokens) {
                    extend_decoder_beam_options(
                        &mut next_by_task[task_idx],
                        state.task,
                        beam,
                        options,
                        5,
                        -1.0,
                    );
                    continue;
                }
                let dynamic_features =
                    state_sequence_dynamic_features(state.task, &beam.tokens, effective_position);
                state
                    .scorer
                    .push_feature_row(&dynamic_features, &mut feature_rows);
                active.push((task_idx, beam.clone(), effective_position));
            }
        }
        if active.is_empty() {
            if next_by_task.iter().all(Vec::is_empty) {
                break;
            }
        } else {
            let features = Tensor::new(active.len(), readout.input_dim, feature_rows).ok()?;
            let topk_limit = max_branch_width
                .saturating_mul(cuda_state_sequence_decode_oversample())
                .clamp(
                    max_branch_width.max(1),
                    vocab.id_to_token.len().min(256).max(1),
                );
            let ranked_rows = symliquid_cuda::readout_cuda::linear_readout_topk_log_probs_cuda(
                &features,
                &readout,
                topk_limit,
                cuda_state_sequence_decode_top_p(),
            )
            .ok()?;
            DECODER_COMPLETION_CUDA_STATE_SEQUENCE_MULTITASK_BATCHES
                .fetch_add(1, Ordering::Relaxed);
            DECODER_COMPLETION_CUDA_STATE_SEQUENCE_MULTITASK_ROWS
                .fetch_add(active.len(), Ordering::Relaxed);
            DECODER_COMPLETION_CUDA_STATE_SEQUENCE_BATCHES.fetch_add(1, Ordering::Relaxed);
            DECODER_COMPLETION_CUDA_STATE_SEQUENCE_ROWS.fetch_add(active.len(), Ordering::Relaxed);
            for (row_idx, (task_idx, beam, effective_position)) in active.into_iter().enumerate() {
                let state = &states[task_idx];
                let Some(ranked_ids) = ranked_rows.get(row_idx) else {
                    continue;
                };
                let options = state_sequence_token_options_from_ranked_ids(
                    state.task,
                    body_ngram,
                    vocab,
                    ranked_ids,
                    &beam.tokens,
                    &state.prompt_tokens,
                    effective_position,
                    state.branch_width,
                );
                extend_decoder_beam_options(
                    &mut next_by_task[task_idx],
                    state.task,
                    &beam,
                    options,
                    5,
                    -1.0,
                );
            }
        }
        let mut all_finished = true;
        for (task_idx, state) in states.iter_mut().enumerate() {
            if state.early_out.is_some() {
                continue;
            }
            let mut next = std::mem::take(&mut next_by_task[task_idx]);
            if next.is_empty() {
                next = state.beams.clone();
            }
            next.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        stable_hash_u64(&format!(
                            "state-seq-batch:{}:{}:{:?}",
                            seed, state.task.task_id, a.tokens
                        ))
                        .cmp(&stable_hash_u64(&format!(
                            "state-seq-batch:{}:{}:{:?}",
                            seed, state.task.task_id, b.tokens
                        )))
                    })
            });
            state.beams = next.into_iter().take(state.beam_width).collect();
            if !state.beams.iter().all(|beam| beam.finished) {
                all_finished = false;
            }
        }
        if all_finished {
            break;
        }
    }
    let mut out = HashMap::new();
    for mut state in states {
        if let Some(rows) = state.early_out {
            out.insert(state.task.task_id.clone(), rows);
            continue;
        }
        let mut rows = Vec::new();
        let mut seen = HashSet::new();
        for candidate_body in state.seeded_completed {
            if state_sequence_candidate_body_ok(state.task, &candidate_body)
                && seen.insert(candidate_body.clone())
            {
                rows.push(candidate_body);
            }
            if rows.len() >= limit {
                break;
            }
        }
        state.beams.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for beam in state.beams {
            let body = join_body_tokens(&beam.tokens);
            for candidate_body in state_sequence_body_variants(state.task, &body) {
                if state_sequence_candidate_body_ok(state.task, &candidate_body)
                    && seen.insert(candidate_body.clone())
                {
                    rows.push(candidate_body);
                }
                if rows.len() >= limit {
                    break;
                }
            }
            if rows.len() >= limit {
                break;
            }
        }
        out.insert(state.task.task_id.clone(), rows);
    }
    Some(out)
}

fn state_sequence_bodies_uncached(
    task: &CodeTask,
    decoder: &StateSequenceDecoder,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    seed: u64,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> Vec<String> {
    if decoder.output_dim == 0 {
        return Vec::new();
    }
    let prompt = prompt_tokens_with_sts(task, sts_streams);
    let static_features = cached_state_sequence_static_features(task, &prompt);
    let static_scores = state_sequence_scores(decoder, static_features.as_slice());
    let cuda_scorer = state_sequence_cuda_scorer(decoder, static_features.as_slice());
    let prompt_tokens = prompt.iter().cloned().collect::<Vec<_>>();
    let mut beams = vec![BeamState {
        tokens: Vec::new(),
        prev2: "<BOS>".to_string(),
        prev1: "<BOS>".to_string(),
        score: 0.0,
        finished: false,
    }];
    let low_latency_fanout = low_latency_candidate_fanout_enabled() && limit <= 8;
    let beam_width = if low_latency_fanout {
        limit.clamp(1, 2)
    } else if execution_shaped_category(&task.category) {
        limit.clamp(2, 4)
    } else if task.split == "public_calibration" {
        limit.clamp(2, 5)
    } else {
        limit.clamp(2, 5)
    };
    let branch_width = if low_latency_fanout {
        vocab.id_to_token.len().min(6)
    } else if execution_shaped_category(&task.category) {
        vocab.id_to_token.len().min(8)
    } else if task.split == "public_calibration" {
        vocab.id_to_token.len().min(12)
    } else {
        vocab.id_to_token.len().min(10)
    };
    let seed_prefixes = if template_free_student_candidates_enabled() {
        contract_decode_seed_prefixes(task)
    } else {
        state_sequence_seed_prefixes(task)
    };
    for prefix in seed_prefixes {
        if !prefix
            .iter()
            .all(|token| vocab.token_to_id.contains_key(token))
        {
            continue;
        }
        if !prefix_is_token_allowed(&prefix) {
            continue;
        }
        let prev1 = prefix
            .last()
            .cloned()
            .unwrap_or_else(|| "<BOS>".to_string());
        let prev2 = prefix
            .iter()
            .rev()
            .nth(1)
            .cloned()
            .unwrap_or_else(|| "<BOS>".to_string());
        beams.push(BeamState {
            tokens: prefix,
            prev2,
            prev1,
            score: if template_free_student_candidates_enabled() {
                1.15
            } else {
                0.85
            },
            finished: false,
        });
    }
    let mut seeded_completed = Vec::new();
    for beam in beams.iter().skip(1) {
        let prefix_body = join_body_tokens(&beam.tokens);
        seeded_completed.extend(state_sequence_body_variants(task, &prefix_body));
    }
    if !seeded_completed.is_empty() {
        let mut seeded_out = Vec::new();
        let mut seen = HashSet::new();
        for candidate_body in &seeded_completed {
            if state_sequence_candidate_body_ok(task, candidate_body)
                && seen.insert(candidate_body.clone())
            {
                seeded_out.push(candidate_body.clone());
            }
            if seeded_out.len() >= limit {
                return seeded_out;
            }
        }
    }
    let max_steps = learned_token_max_steps(
        task,
        if low_latency_fanout {
            28
        } else if task.split == "public_calibration" {
            56
        } else {
            48
        },
    );
    for _step in 0..max_steps {
        let mut next = Vec::new();
        let mut active_beams = Vec::new();
        let mut feature_rows = Vec::new();
        let mut positions = Vec::new();
        for beam in &beams {
            if beam.finished {
                next.push(beam.clone());
                continue;
            }
            let effective_position = learned_position_cap(&task.category, beam.tokens.len());
            if let Some(options) = forced_block_token_options(&beam.tokens) {
                extend_decoder_beam_options(&mut next, task, beam, options, 5, -1.0);
                continue;
            }
            if let Some(scorer) = cuda_scorer.as_ref() {
                let dynamic_features =
                    state_sequence_dynamic_features(task, &beam.tokens, effective_position);
                scorer.push_feature_row(&dynamic_features, &mut feature_rows);
                positions.push(effective_position);
                active_beams.push(beam);
                continue;
            }
            let dynamic_features =
                state_sequence_dynamic_features(task, &beam.tokens, effective_position);
            let scores =
                state_sequence_scores_with_base(&static_scores, decoder, &dynamic_features);
            let options = state_sequence_token_options_from_scores(
                task,
                body_ngram,
                vocab,
                &scores,
                &beam.tokens,
                &prompt_tokens,
                effective_position,
                branch_width,
            );
            if options.is_empty() {
                continue;
            }
            extend_decoder_beam_options(
                &mut next,
                task,
                beam,
                options.into_iter().collect(),
                5,
                -1.0,
            );
        }
        if !active_beams.is_empty() {
            let mut used_cuda = false;
            if let Some(scorer) = cuda_scorer.as_ref() {
                if let Ok(features) =
                    Tensor::new(active_beams.len(), scorer.input_dim(), feature_rows)
                {
                    if let Some(token_options) = state_sequence_cuda_token_options_batch(
                        task,
                        body_ngram,
                        vocab,
                        scorer,
                        &features,
                        &active_beams,
                        &prompt_tokens,
                        &positions,
                        branch_width,
                    ) {
                        used_cuda = true;
                        for (row_idx, beam) in active_beams.iter().copied().enumerate() {
                            for (token, score) in token_options
                                .get(row_idx)
                                .into_iter()
                                .flat_map(|row| row.iter().cloned())
                            {
                                extend_decoder_beam_options(
                                    &mut next,
                                    task,
                                    beam,
                                    vec![(token, score)],
                                    5,
                                    -1.0,
                                );
                            }
                        }
                    }
                }
            }
            if !used_cuda {
                DECODER_COMPLETION_CUDA_STATE_SEQUENCE_FALLBACK_ROWS
                    .fetch_add(active_beams.len(), Ordering::Relaxed);
                for beam in active_beams {
                    let effective_position =
                        learned_position_cap(&task.category, beam.tokens.len());
                    let dynamic_features =
                        state_sequence_dynamic_features(task, &beam.tokens, effective_position);
                    let scores =
                        state_sequence_scores_with_base(&static_scores, decoder, &dynamic_features);
                    let options = state_sequence_token_options_from_scores(
                        task,
                        body_ngram,
                        vocab,
                        &scores,
                        &beam.tokens,
                        &prompt_tokens,
                        effective_position,
                        branch_width,
                    );
                    if options.is_empty() {
                        continue;
                    }
                    extend_decoder_beam_options(
                        &mut next,
                        task,
                        beam,
                        options.into_iter().collect(),
                        5,
                        -1.0,
                    );
                }
            }
        }
        if next.is_empty() {
            break;
        }
        next.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    stable_hash_u64(&format!(
                        "state-seq:{}:{}:{:?}",
                        seed, task.task_id, a.tokens
                    ))
                    .cmp(&stable_hash_u64(&format!(
                        "state-seq:{}:{}:{:?}",
                        seed, task.task_id, b.tokens
                    )))
                })
        });
        beams = next.into_iter().take(beam_width).collect();
        if beams.iter().all(|beam| beam.finished) {
            break;
        }
    }
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for candidate_body in seeded_completed {
        if state_sequence_candidate_body_ok(task, &candidate_body)
            && seen.insert(candidate_body.clone())
        {
            out.push(candidate_body);
        }
        if out.len() >= limit {
            break;
        }
    }
    beams.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for beam in beams {
        let body = join_body_tokens(&beam.tokens);
        for candidate_body in state_sequence_body_variants(task, &body) {
            if state_sequence_candidate_body_ok(task, &candidate_body)
                && seen.insert(candidate_body.clone())
            {
                out.push(candidate_body);
            }
            if out.len() >= limit {
                break;
            }
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}
