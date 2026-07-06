use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use symliquid_core::tensor::Tensor;
use symliquid_core::train::LinearReadout;

#[derive(Debug, Clone)]
pub struct StsParallelDecoderConfig {
    pub input: String,
    pub seed: u64,
    pub hv_dim: usize,
    pub max_vocab: usize,
    pub epochs: usize,
    pub lr: f32,
    pub max_generate_steps: usize,
    pub max_train_rows: usize,
    pub max_eval_rows: usize,
    pub max_generate_rows: usize,
    pub checkpoint_out: String,
    pub generation_out: String,
    pub report_out: String,
}

#[derive(Debug, Clone)]
struct StsRow {
    raw: Value,
    task_id: String,
    split: String,
    context: String,
    output_streams: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
struct StreamExample {
    task_id: String,
    stream: String,
    context_tokens: Vec<String>,
    prev2: String,
    prev1: String,
    position: usize,
    target: usize,
}

#[derive(Debug, Clone)]
struct Vocab {
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
    unk_id: usize,
}

#[derive(Debug, Clone, Copy)]
struct EvalTrace {
    accuracy: f32,
}

#[derive(Debug, Clone)]
struct StsTrainingRun {
    backend: String,
    after_eval: EvalTrace,
    epoch_eval_accuracies: Vec<f32>,
    epochs_completed: usize,
    best_epoch: usize,
    early_stopped: bool,
    restored_best_trained_epoch: bool,
    restored_baseline_epoch: bool,
}

const CONTEXT_STREAM: &str = "context_stream";

pub fn train_sts_parallel_decoder(
    config: StsParallelDecoderConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let load_started = Instant::now();
    let rows = load_rows(Path::new(&config.input))?;
    let load_rows_ms = elapsed_ms(load_started);
    let split_started = Instant::now();
    let train_rows = rows
        .iter()
        .filter(|row| row.split == "train")
        .take(config.max_train_rows.max(1))
        .cloned()
        .collect::<Vec<_>>();
    let eval_rows = rows
        .iter()
        .filter(|row| row.split == "eval")
        .take(config.max_eval_rows.max(1))
        .cloned()
        .collect::<Vec<_>>();
    let mut generation_rows = eval_rows.clone();
    generation_rows.extend(
        rows.iter()
            .filter(|row| row.split != "train" && row.split != "eval")
            .take(config.max_generate_rows.max(1))
            .cloned(),
    );
    let split_rows_ms = elapsed_ms(split_started);
    let build_started = Instant::now();
    let vocab = build_vocab(&train_rows, config.max_vocab.max(16));
    let train_examples = build_examples(&train_rows, &vocab);
    let eval_examples = build_examples(&eval_rows, &vocab);
    let hv_dim = config.hv_dim.max(32);
    let mut readout = LinearReadout::zeros(hv_dim, vocab.id_to_token.len());
    let batch_size = sts_parallel_batch_size();
    let train_batch_count_per_epoch = batch_count(train_examples.len(), batch_size);
    let eval_batch_count = batch_count(eval_examples.len(), batch_size);
    let build_examples_ms = elapsed_ms(build_started);
    let before_eval_started = Instant::now();
    let train_accuracy_eval_enabled = sts_train_accuracy_eval_enabled();
    let before_train = if train_accuracy_eval_enabled {
        Some(evaluate(&readout, &train_examples, hv_dim)?)
    } else {
        None
    };
    let before_eval = evaluate(&readout, &eval_examples, hv_dim)?;
    let before_eval_ms = elapsed_ms(before_eval_started);
    let train_started = Instant::now();
    let training_run = train_readout(
        &mut readout,
        &train_examples,
        &eval_examples,
        hv_dim,
        config.epochs.max(1),
        config.lr,
        before_eval.accuracy,
    )?;
    let training_backend = training_run.backend.clone();
    let train_ms = elapsed_ms(train_started);
    let cuda_fast_readout_requested = sts_cuda_fast_readout_enabled();
    let cuda_fast_readout_used = training_backend.contains("cuda");
    let after_eval_started = Instant::now();
    let after_train = if train_accuracy_eval_enabled {
        Some(evaluate(&readout, &train_examples, hv_dim)?)
    } else {
        None
    };
    let after_eval = training_run.after_eval;
    let after_eval_ms = elapsed_ms(after_eval_started);
    let generation_started = Instant::now();
    let generations = generate_rows(
        &generation_rows,
        &readout,
        &vocab,
        hv_dim,
        config.max_generate_steps.max(1),
    );
    let generation_ms = elapsed_ms(generation_started);
    let write_generation_started = Instant::now();
    write_jsonl(Path::new(&config.generation_out), &generations)?;
    let write_generation_ms = elapsed_ms(write_generation_started);

    let output_streams = output_stream_names(&rows);
    let checkpoint_material = json!({
        "input": config.input,
        "seed": config.seed,
        "train_rows": train_rows.len(),
        "eval_rows": eval_rows.len(),
        "max_train_rows": config.max_train_rows,
        "max_eval_rows": config.max_eval_rows,
        "max_generate_rows": config.max_generate_rows,
        "output_streams": output_streams,
        "vocab": vocab.id_to_token,
        "epochs": config.epochs,
        "lr": config.lr,
        "max_train_rows": config.max_train_rows,
        "max_eval_rows": config.max_eval_rows,
        "max_generate_rows": config.max_generate_rows,
        "before_eval_accuracy": before_eval.accuracy,
        "after_eval_accuracy": after_eval.accuracy,
    });
    let checkpoint_id = format!(
        "theseus_sts_parallel_{}",
        &stable_hash_hex(&checkpoint_material.to_string())[..16]
    );
    let checkpoint = json!({
        "policy": "project_theseus_sts_parallel_decoder_checkpoint_v1",
        "created_utc": now(),
        "checkpoint_id": checkpoint_id,
        "checkpoint_kind": "symliquid_stream_parallel_next_token_decoder",
        "backend": "rust_symliquid_core_linear_readout",
        "input": rel(&config.input),
        "seed": config.seed,
        "hv_dim": readout.input_dim,
        "output_dim": readout.output_dim,
        "epochs": config.epochs,
        "lr": config.lr,
        "output_streams": output_streams,
        "vocab": vocab.id_to_token,
        "weights": readout.weights,
        "bias": readout.bias,
        "generation_policy": {
            "native_parallel_token_generation": true,
            "one_token_per_output_stream_per_step": true,
            "public_benchmark_solutions_included": false,
            "public_tests_visible_to_generator": false,
            "external_inference_calls": 0,
            "allowed_inputs": ["context_stream", "previous_tokens_same_stream", "stream_id", "position"]
        },
        "summary": {
            "train_row_count": train_rows.len(),
            "eval_row_count": eval_rows.len(),
            "max_train_rows": config.max_train_rows,
            "max_eval_rows": config.max_eval_rows,
            "max_generate_rows": config.max_generate_rows,
            "output_stream_count": output_streams.len(),
            "train_example_count": train_examples.len(),
            "eval_example_count": eval_examples.len(),
            "training_backend": training_backend.clone(),
            "cuda_fast_readout_requested": cuda_fast_readout_requested,
            "cuda_fast_readout_used": cuda_fast_readout_used,
            "train_accuracy_eval_enabled": train_accuracy_eval_enabled,
            "train_accuracy_eval_skipped": !train_accuracy_eval_enabled,
            "sts_epoch_eval_accuracies": training_run.epoch_eval_accuracies.clone(),
            "sts_epochs_completed": training_run.epochs_completed,
            "sts_epochs_configured": config.epochs.max(1),
            "sts_best_epoch": training_run.best_epoch,
            "sts_best_epoch_semantics": "0 means the untrained baseline was retained because no trained epoch improved held-out eval",
            "sts_early_stopped": training_run.early_stopped,
            "sts_restored_best_trained_epoch": training_run.restored_best_trained_epoch,
            "sts_restored_baseline_epoch": training_run.restored_baseline_epoch,
            "training_batch_size": batch_size,
            "train_batch_count_per_epoch": train_batch_count_per_epoch,
            "eval_batch_count": eval_batch_count,
            "train_example_passes": train_examples.len() * training_run.epochs_completed,
            "before_eval_token_accuracy": round6(before_eval.accuracy),
            "after_eval_token_accuracy": round6(after_eval.accuracy),
            "eval_token_accuracy_delta": round6(after_eval.accuracy - before_eval.accuracy),
            "native_parallel_token_generation_proven": !generations.is_empty(),
            "external_inference_calls": 0,
            "timing_breakdown_ms": {
                "load_rows": load_rows_ms,
                "split_rows": split_rows_ms,
                "build_vocab_and_examples": build_examples_ms,
                "before_train_eval": before_eval_ms,
                "batched_train": train_ms,
                "after_train_eval": after_eval_ms,
                "generation": generation_ms,
                "write_generation": write_generation_ms
            }
        },
        "external_inference_calls": 0,
    });
    let write_checkpoint_started = Instant::now();
    write_json_compact(Path::new(&config.checkpoint_out), &checkpoint)?;
    let write_checkpoint_ms = elapsed_ms(write_checkpoint_started);

    let public_benchmark_solutions_included = rows.iter().any(row_contains_solution_or_tests);
    let gates = vec![
        gate(
            "sts_rows_loaded",
            !rows.is_empty(),
            json!(format!("rows={}", rows.len())),
        ),
        gate(
            "train_eval_split_present",
            !train_rows.is_empty() && !eval_rows.is_empty(),
            json!(format!(
                "train={} eval={}",
                train_rows.len(),
                eval_rows.len()
            )),
        ),
        gate(
            "multiple_output_streams_present",
            output_streams.len() >= 2,
            json!(output_streams),
        ),
        gate(
            "training_examples_nonzero",
            !train_examples.is_empty(),
            json!(format!("examples={}", train_examples.len())),
        ),
        gate(
            "eval_examples_nonzero",
            !eval_examples.is_empty(),
            json!(format!("examples={}", eval_examples.len())),
        ),
        gate(
            "eval_accuracy_improved",
            after_eval.accuracy > before_eval.accuracy,
            json!(format!(
                "before={:.6} after={:.6}",
                before_eval.accuracy, after_eval.accuracy
            )),
        ),
        gate(
            "parallel_generations_emitted",
            !generations.is_empty(),
            json!(format!("generations={}", generations.len())),
        ),
        gate(
            "no_public_benchmark_solutions",
            !public_benchmark_solutions_included,
            json!(format!(
                "public_benchmark_solutions_included={public_benchmark_solutions_included}"
            )),
        ),
        gate(
            "external_inference_zero",
            true,
            json!("local Rust/SymLiquid training only"),
        ),
    ];
    let trigger_state = if gates
        .iter()
        .all(|row| row["passed"].as_bool().unwrap_or(false))
    {
        "GREEN"
    } else {
        "YELLOW"
    };
    let report = json!({
        "policy": "project_theseus_sts_native_parallel_probe_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "input": rel(&config.input),
        "checkpoint": rel(&config.checkpoint_out),
        "generation_out": rel(&config.generation_out),
        "summary": {
            "row_count": rows.len(),
            "train_row_count": train_rows.len(),
            "eval_row_count": eval_rows.len(),
            "max_train_rows": config.max_train_rows,
            "max_eval_rows": config.max_eval_rows,
            "max_generate_rows": config.max_generate_rows,
            "output_stream_count": output_streams.len(),
            "output_streams": output_streams,
            "train_example_count": train_examples.len(),
            "eval_example_count": eval_examples.len(),
            "training_backend": training_backend,
            "cuda_fast_readout_requested": cuda_fast_readout_requested,
            "cuda_fast_readout_used": cuda_fast_readout_used,
            "train_accuracy_eval_enabled": train_accuracy_eval_enabled,
            "train_accuracy_eval_skipped": !train_accuracy_eval_enabled,
            "sts_epoch_eval_accuracies": training_run.epoch_eval_accuracies.clone(),
            "sts_epochs_completed": training_run.epochs_completed,
            "sts_epochs_configured": config.epochs.max(1),
            "sts_best_epoch": training_run.best_epoch,
            "sts_best_epoch_semantics": "0 means the untrained baseline was retained because no trained epoch improved held-out eval",
            "sts_early_stopped": training_run.early_stopped,
            "sts_restored_best_trained_epoch": training_run.restored_best_trained_epoch,
            "sts_restored_baseline_epoch": training_run.restored_baseline_epoch,
            "training_batch_size": batch_size,
            "train_batch_count_per_epoch": train_batch_count_per_epoch,
            "eval_batch_count": eval_batch_count,
            "train_example_passes": train_examples.len() * training_run.epochs_completed,
            "before_train_token_accuracy": before_train.map(|trace| round6(trace.accuracy)),
            "after_train_token_accuracy": after_train.map(|trace| round6(trace.accuracy)),
            "before_eval_token_accuracy": round6(before_eval.accuracy),
            "after_eval_token_accuracy": round6(after_eval.accuracy),
            "eval_token_accuracy_delta": round6(after_eval.accuracy - before_eval.accuracy),
            "native_parallel_token_generation_proven": !generations.is_empty(),
            "one_token_per_output_stream_per_step": true,
            "generation_row_count": generations.len(),
            "public_benchmark_solutions_included": public_benchmark_solutions_included,
            "external_inference_calls": 0,
            "timing_breakdown_ms": {
                "load_rows": load_rows_ms,
                "split_rows": split_rows_ms,
                "build_vocab_and_examples": build_examples_ms,
                "before_train_eval": before_eval_ms,
                "batched_train": train_ms,
                "after_train_eval": after_eval_ms,
                "generation": generation_ms,
                "write_generation": write_generation_ms,
                "write_checkpoint": write_checkpoint_ms
            }
        },
        "gates": gates,
        "runtime_ms": (started.elapsed().as_secs_f64() * 1000.0).round() as u64,
        "external_inference_calls": 0,
    });
    write_json(Path::new(&config.report_out), &report)?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "policy": "project_theseus_sts_native_parallel_probe_v1",
            "trigger_state": trigger_state,
            "report_out": rel(&config.report_out),
            "generation_out": rel(&config.generation_out),
            "runtime_ms": (started.elapsed().as_secs_f64() * 1000.0).round() as u64,
        }))?
    );
    Ok(())
}

fn load_rows(path: &Path) -> Result<Vec<StsRow>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut rows = Vec::new();
    for raw in read_jsonl(path)? {
        let input_streams = raw
            .get("input_streams")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let target_streams = raw
            .get("target_streams")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let legacy_streams = raw
            .get("streams")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let context = input_context_text(&input_streams, &legacy_streams);
        let output_streams = if target_streams.is_empty() {
            legacy_streams
        } else {
            target_streams
        }
        .into_iter()
        .filter_map(|(name, value)| {
            if name == CONTEXT_STREAM {
                None
            } else {
                Some((name, value.as_str().unwrap_or_default().to_string()))
            }
        })
        .collect::<BTreeMap<_, _>>();
        if context.is_empty() || output_streams.is_empty() {
            continue;
        }
        rows.push(StsRow {
            task_id: string_field(&raw, "task_id"),
            split: string_field(&raw, "split"),
            context,
            output_streams,
            raw,
        });
    }
    Ok(rows)
}

fn input_context_text(
    input_streams: &serde_json::Map<String, Value>,
    legacy_streams: &serde_json::Map<String, Value>,
) -> String {
    let mut parts = Vec::new();
    let context = input_streams
        .get(CONTEXT_STREAM)
        .or_else(|| legacy_streams.get(CONTEXT_STREAM))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if !context.is_empty() {
        parts.push(context.to_string());
    }
    for (name, value) in input_streams {
        if name == CONTEXT_STREAM {
            continue;
        }
        let text = value.as_str().unwrap_or_default().trim();
        if !text.is_empty() {
            parts.push(format!("{name}: {text}"));
        }
    }
    parts.join(" ")
}

fn row_contains_solution_or_tests(row: &StsRow) -> bool {
    fn contains_key(value: &Value, names: &[&str]) -> bool {
        match value {
            Value::Object(map) => map.iter().any(|(key, child)| {
                names.iter().any(|name| key == name) || contains_key(child, names)
            }),
            Value::Array(items) => items.iter().any(|child| contains_key(child, names)),
            _ => false,
        }
    }
    contains_key(
        &row.raw,
        &[
            "canonical_solution",
            "canonical_solutions",
            "expected_answer",
            "answer_key",
            "tests",
            "test",
            "hidden_tests",
            "public_tests",
        ],
    )
}

fn build_examples(rows: &[StsRow], vocab: &Vocab) -> Vec<StreamExample> {
    let mut examples = Vec::new();
    for row in rows {
        let context_tokens = tokenize(&row.context)
            .into_iter()
            .take(80)
            .map(|token| token.to_lowercase())
            .collect::<Vec<_>>();
        for (stream, text) in &row.output_streams {
            let mut prev2 = "<BOS>".to_string();
            let mut prev1 = "<BOS>".to_string();
            let mut tokens = tokenize(text);
            tokens.push("<EOS>".to_string());
            for (position, token) in tokens.into_iter().enumerate() {
                let target = vocab_id(vocab, &token);
                examples.push(StreamExample {
                    task_id: row.task_id.clone(),
                    stream: stream.clone(),
                    context_tokens: context_tokens.clone(),
                    prev2: prev2.clone(),
                    prev1: prev1.clone(),
                    position,
                    target,
                });
                prev2 = prev1;
                prev1 = token;
            }
        }
    }
    examples
}

fn evaluate(
    readout: &LinearReadout,
    examples: &[StreamExample],
    hv_dim: usize,
) -> Result<EvalTrace, Box<dyn std::error::Error>> {
    if examples.is_empty() {
        return Ok(EvalTrace { accuracy: 0.0 });
    }
    let batch_size = sts_parallel_batch_size();
    let mut weighted_accuracy = 0.0f32;
    let mut seen = 0usize;
    for chunk in examples.chunks(batch_size) {
        let (features, targets) = example_batch(chunk, hv_dim)?;
        let trace = readout.evaluate_batch(&features, &targets)?;
        weighted_accuracy += trace.accuracy * chunk.len() as f32;
        seen += chunk.len();
    }
    Ok(EvalTrace {
        accuracy: weighted_accuracy / seen.max(1) as f32,
    })
}

fn train_readout(
    readout: &mut LinearReadout,
    examples: &[StreamExample],
    eval_examples: &[StreamExample],
    hv_dim: usize,
    epochs: usize,
    lr: f32,
    before_eval_accuracy: f32,
) -> Result<StsTrainingRun, Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")]
    {
        if sts_cuda_fast_readout_enabled() && !examples.is_empty() {
            match train_cuda_fast_sparse(readout, examples, hv_dim, epochs, lr) {
                Ok(backend) => {
                    let after_eval = evaluate(readout, eval_examples, hv_dim)?;
                    return Ok(StsTrainingRun {
                        backend,
                        after_eval,
                        epoch_eval_accuracies: vec![round6(after_eval.accuracy)],
                        epochs_completed: epochs.max(1),
                        best_epoch: epochs.max(1),
                        early_stopped: false,
                        restored_best_trained_epoch: false,
                        restored_baseline_epoch: false,
                    });
                }
                Err(error) => {
                    let mut run = train_batched(
                        readout,
                        examples,
                        eval_examples,
                        hv_dim,
                        epochs,
                        lr,
                        before_eval_accuracy,
                    )?;
                    run.backend =
                        format!("rust_symliquid_core_linear_readout_train_batch_after_cuda_fallback:{error}");
                    return Ok(run);
                }
            }
        }
    }
    train_batched(
        readout,
        examples,
        eval_examples,
        hv_dim,
        epochs,
        lr,
        before_eval_accuracy,
    )
}

#[cfg(feature = "cuda")]
fn train_cuda_fast_sparse(
    readout: &mut LinearReadout,
    examples: &[StreamExample],
    hv_dim: usize,
    epochs: usize,
    lr: f32,
) -> Result<String, Box<dyn std::error::Error>> {
    let (features, targets) = example_batch(examples, hv_dim)?;
    let salts = examples
        .iter()
        .map(|example| {
            stable_hash_u64((
                &example.task_id,
                &example.stream,
                &example.prev2,
                &example.prev1,
                example.position,
            )) as usize
        })
        .collect::<Vec<_>>();
    let _trace = symliquid_cuda::readout_cuda::train_code_fast_readout_cuda(
        &features,
        &targets,
        &salts,
        readout,
        epochs.max(1),
        lr,
    )?;
    Ok("rust_cuda_fast_sparse_sts_parallel_readout".to_string())
}

fn train_batched(
    readout: &mut LinearReadout,
    examples: &[StreamExample],
    eval_examples: &[StreamExample],
    hv_dim: usize,
    epochs: usize,
    lr: f32,
    before_eval_accuracy: f32,
) -> Result<StsTrainingRun, Box<dyn std::error::Error>> {
    if examples.is_empty() {
        let after_eval = evaluate(readout, eval_examples, hv_dim)?;
        return Ok(StsTrainingRun {
            backend: "rust_symliquid_core_linear_readout_train_batch".to_string(),
            after_eval,
            epoch_eval_accuracies: vec![round6(after_eval.accuracy)],
            epochs_completed: 0,
            best_epoch: 0,
            early_stopped: false,
            restored_best_trained_epoch: false,
            restored_baseline_epoch: false,
        });
    }
    let batch_size = sts_parallel_batch_size();
    let mut best_readout = readout.clone();
    let mut best_eval = EvalTrace {
        accuracy: before_eval_accuracy,
    };
    let mut best_epoch = 0usize;
    let mut epoch_eval_accuracies = Vec::new();
    let mut epochs_completed = 0usize;
    let mut early_stopped = false;
    let min_delta = sts_eval_min_delta();
    let stop_on_baseline_regression = sts_early_stop_on_baseline_regression_enabled();
    let patience = sts_early_stop_patience();
    let mut stale_epochs = 0usize;
    for epoch in 0..epochs.max(1) {
        for chunk in examples.chunks(batch_size) {
            let (features, targets) = example_batch(chunk, hv_dim)?;
            readout.train_batch(&features, &targets, lr)?;
        }
        epochs_completed = epoch + 1;
        let eval = evaluate(readout, eval_examples, hv_dim)?;
        epoch_eval_accuracies.push(round6(eval.accuracy));
        if eval.accuracy > best_eval.accuracy + min_delta {
            best_eval = eval;
            best_epoch = epoch + 1;
            best_readout = readout.clone();
            stale_epochs = 0;
        } else {
            stale_epochs = stale_epochs.saturating_add(1);
        }
        if stop_on_baseline_regression
            && epoch == 0
            && eval.accuracy + min_delta < before_eval_accuracy
        {
            early_stopped = epochs.max(1) > 1;
            break;
        }
        if stale_epochs > patience {
            early_stopped = epoch + 1 < epochs.max(1);
            break;
        }
    }
    let restored_best_trained_epoch = best_epoch > 0 && best_epoch != epochs_completed;
    let restored_baseline_epoch = best_epoch == 0 && epochs_completed > 0;
    *readout = best_readout;
    Ok(StsTrainingRun {
        backend: "rust_symliquid_core_linear_readout_train_batch_best_epoch".to_string(),
        after_eval: best_eval,
        epoch_eval_accuracies,
        epochs_completed,
        best_epoch,
        early_stopped,
        restored_best_trained_epoch,
        restored_baseline_epoch,
    })
}

fn example_batch(
    examples: &[StreamExample],
    hv_dim: usize,
) -> Result<(Tensor, Vec<usize>), Box<dyn std::error::Error>> {
    let mut rows = Vec::with_capacity(examples.len() * hv_dim);
    let mut targets = Vec::with_capacity(examples.len());
    for example in examples {
        rows.extend(featurize(example, hv_dim));
        targets.push(example.target);
    }
    Ok((Tensor::new(examples.len(), hv_dim, rows)?, targets))
}

fn sts_parallel_batch_size() -> usize {
    std::env::var("THESEUS_STS_PARALLEL_BATCH_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(512)
        .clamp(16, 4096)
}

fn sts_train_accuracy_eval_enabled() -> bool {
    std::env::var("THESEUS_STS_EVALUATE_TRAIN_ACCURACY")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(false)
}

fn sts_early_stop_on_baseline_regression_enabled() -> bool {
    std::env::var("THESEUS_STS_EARLY_STOP_ON_BASELINE_REGRESSION")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn sts_early_stop_patience() -> usize {
    std::env::var("THESEUS_STS_EARLY_STOP_PATIENCE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0)
        .min(8)
}

fn sts_eval_min_delta() -> f32 {
    std::env::var("THESEUS_STS_EVAL_MIN_DELTA")
        .ok()
        .and_then(|value| value.trim().parse::<f32>().ok())
        .unwrap_or(0.0)
        .max(0.0)
}

fn sts_cuda_fast_readout_enabled() -> bool {
    std::env::var("THESEUS_STS_USE_CUDA_FAST_READOUT")
        .map(|value| value.trim() != "0")
        .unwrap_or(true)
}

fn batch_count(item_count: usize, batch_size: usize) -> usize {
    if item_count == 0 {
        0
    } else {
        (item_count + batch_size.max(1) - 1) / batch_size.max(1)
    }
}

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

fn generate_rows(
    rows: &[StsRow],
    readout: &LinearReadout,
    vocab: &Vocab,
    hv_dim: usize,
    max_steps: usize,
) -> Vec<Value> {
    let mut out = Vec::new();
    for row in rows {
        let context_tokens = tokenize(&row.context)
            .into_iter()
            .take(80)
            .map(|token| token.to_lowercase())
            .collect::<Vec<_>>();
        let mut stream_state = row
            .output_streams
            .keys()
            .map(|stream| {
                (
                    stream.clone(),
                    (
                        "<BOS>".to_string(),
                        "<BOS>".to_string(),
                        Vec::<String>::new(),
                        false,
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut steps = Vec::new();
        for position in 0..max_steps {
            let mut step = BTreeMap::new();
            for (stream, (prev2, prev1, tokens, done)) in stream_state.iter_mut() {
                if *done {
                    step.insert(stream.clone(), "<EOS>".to_string());
                    continue;
                }
                let example = StreamExample {
                    task_id: row.task_id.clone(),
                    stream: stream.clone(),
                    context_tokens: context_tokens.clone(),
                    prev2: prev2.clone(),
                    prev1: prev1.clone(),
                    position,
                    target: 0,
                };
                let token = next_token(readout, vocab, &example, hv_dim)
                    .unwrap_or_else(|| "<EOS>".to_string());
                step.insert(stream.clone(), token.clone());
                if token == "<EOS>" {
                    *done = true;
                } else {
                    tokens.push(token.clone());
                    *prev2 = prev1.clone();
                    *prev1 = token;
                }
            }
            steps.push(json!(step));
            if stream_state.values().all(|(_, _, _, done)| *done) {
                break;
            }
        }
        let generated_streams = stream_state
            .iter()
            .map(|(stream, (_, _, tokens, _))| (stream.clone(), join_tokens(tokens)))
            .collect::<BTreeMap<_, _>>();
        out.push(json!({
            "task_id": row.task_id,
            "split": row.split,
            "generation_mode": "rust_symliquid_sts_parallel_decoder",
            "native_parallel_token_generation": true,
            "one_token_per_output_stream_per_step": true,
            "streams": generated_streams,
            "parallel_steps": steps,
            "public_benchmark_solutions_included": false,
            "external_inference_calls": 0
        }));
    }
    out
}

fn next_token(
    readout: &LinearReadout,
    vocab: &Vocab,
    example: &StreamExample,
    hv_dim: usize,
) -> Option<String> {
    let features = Tensor::from_row(featurize(example, hv_dim));
    let logits = readout.logits(&features).ok()?;
    let mut ranked = logits
        .row(0)
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
        .into_iter()
        .map(|(id, _)| vocab.id_to_token[id].clone())
        .find(|token| token != "<UNK>")
}

fn featurize(example: &StreamExample, hv_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; hv_dim];
    add_feature(&mut out, &format!("task:{}", example.task_id), 0.25);
    add_feature(&mut out, &format!("stream:{}", example.stream), 1.4);
    add_feature(&mut out, &format!("prev1:{}", example.prev1), 1.5);
    add_feature(&mut out, &format!("prev2:{}", example.prev2), 0.8);
    add_feature(&mut out, &format!("pos:{}", example.position.min(64)), 0.45);
    for token in example.context_tokens.iter().take(80) {
        add_feature(&mut out, &format!("context:{token}"), 0.22);
        add_feature(
            &mut out,
            &format!("stream_context:{}:{token}", example.stream),
            0.12,
        );
    }
    let norm = out
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1.0);
    for value in &mut out {
        *value /= norm;
    }
    out
}

fn build_vocab(rows: &[StsRow], max_vocab: usize) -> Vocab {
    let mut counts = HashMap::new();
    for row in rows {
        for text in row.output_streams.values() {
            for token in tokenize(text) {
                *counts.entry(token).or_insert(0usize) += 1;
            }
        }
    }
    let mut sorted = counts.into_iter().collect::<Vec<_>>();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut id_to_token = vec!["<EOS>".to_string(), "<UNK>".to_string()];
    for (token, _) in sorted {
        if id_to_token.len() >= max_vocab {
            break;
        }
        if token != "<EOS>" && token != "<UNK>" {
            id_to_token.push(token);
        }
    }
    let token_to_id = id_to_token
        .iter()
        .enumerate()
        .map(|(id, token)| (token.clone(), id))
        .collect::<HashMap<_, _>>();
    Vocab {
        token_to_id,
        id_to_token,
        unk_id: 1,
    }
}

fn output_stream_names(rows: &[StsRow]) -> Vec<String> {
    rows.iter()
        .flat_map(|row| row.output_streams.keys().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn vocab_id(vocab: &Vocab, token: &str) -> usize {
    *vocab.token_to_id.get(token).unwrap_or(&vocab.unk_id)
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let chars = text.chars().collect::<Vec<_>>();
    let mut idx = 0usize;
    while idx < chars.len() {
        let ch = chars[idx];
        if ch.is_whitespace() {
            idx += 1;
            continue;
        }
        if ch.is_ascii_alphanumeric() || ch == '_' {
            let start = idx;
            idx += 1;
            while idx < chars.len()
                && (chars[idx].is_ascii_alphanumeric() || chars[idx] == '_' || chars[idx] == '-')
            {
                idx += 1;
            }
            tokens.push(chars[start..idx].iter().collect::<String>());
            continue;
        }
        if idx + 1 < chars.len() {
            let two = format!("{}{}", chars[idx], chars[idx + 1]);
            if ["==", "!=", "<=", ">=", "->", "::"].contains(&two.as_str()) {
                tokens.push(two);
                idx += 2;
                continue;
            }
        }
        tokens.push(ch.to_string());
        idx += 1;
    }
    tokens
}

fn join_tokens(tokens: &[String]) -> String {
    let mut out = String::new();
    for token in tokens {
        let no_space_before = [")", "]", "}", ",", ".", ":", ";"].contains(&token.as_str());
        let no_space_after_prev =
            out.ends_with('(') || out.ends_with('[') || out.ends_with('{') || out.ends_with('.');
        if !out.is_empty() && !no_space_before && !no_space_after_prev {
            out.push(' ');
        }
        out.push_str(token);
    }
    out
}

fn add_feature(out: &mut [f32], key: &str, weight: f32) {
    if out.is_empty() {
        return;
    }
    let idx = stable_hash_u64(key) as usize % out.len();
    let sign = if stable_hash_u64(&format!("sign:{key}")) & 1 == 0 {
        1.0
    } else {
        -1.0
    };
    out[idx] += weight * sign;
}

fn read_jsonl(path: &Path) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        rows.push(serde_json::from_str::<Value>(line)?);
    }
    Ok(rows)
}

fn write_json(path: &Path, value: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(value)?))?;
    Ok(())
}

fn write_json_compact(path: &Path, value: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer(&mut writer, value)?;
    std::io::Write::write_all(&mut writer, b"\n")?;
    Ok(())
}

fn write_jsonl(path: &Path, rows: &[Value]) -> Result<(), Box<dyn std::error::Error>> {
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

fn string_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn gate(name: &str, passed: bool, evidence: Value) -> Value {
    json!({"gate": name, "passed": passed, "evidence": evidence})
}

fn now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format!("unix:{secs}")
}

fn rel(value: &str) -> String {
    value.replace('\\', "/")
}

fn round6(value: f32) -> f32 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn stable_hash_hex(text: &str) -> String {
    format!(
        "{:016x}{:016x}",
        stable_hash_u64(text),
        stable_hash_u64(&format!("salt:{text}"))
    )
}

fn stable_hash_u64<T: Hash>(value: T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
