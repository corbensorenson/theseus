use clap::{Parser, Subcommand};
use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};
use theseus_corpus::{
    encode_kerc_global_target, exact_text_tokens, exact_text_tokens_scalar, ids_sha256,
    kerc_code_tokens, ExactVocabulary, KERC_KERNEL_OBJECTIVES,
};

#[derive(Debug, Parser)]
#[command(about = "Exact Project Theseus corpus primitives")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Encode {
        #[arg(long)]
        vocab: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = false)]
        include_tokens: bool,
    },
    Benchmark {
        #[arg(long)]
        vocab: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 3)]
        repetitions: usize,
    },
    ScannerBenchmark {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 3)]
        repetitions: usize,
        #[arg(long, default_value_t = false)]
        scalar: bool,
    },
    Materialize {
        #[arg(long)]
        vocab: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long)]
        target_offset: i32,
        #[arg(long, default_value_t = 512)]
        sequence_length: usize,
        #[arg(long, default_value_t = 1)]
        workers: usize,
        #[arg(long, default_value_t = 256)]
        chunk_rows: usize,
    },
    CacheMaterialize {
        #[arg(long)]
        vocab: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        cache_root: PathBuf,
        #[arg(long)]
        target_offset: i32,
        #[arg(long, default_value_t = 512)]
        sequence_length: usize,
        #[arg(long, default_value_t = 1)]
        workers: usize,
        #[arg(long, default_value_t = 256)]
        chunk_rows: usize,
        #[arg(long, default_value_t = 8_589_934_592_u64)]
        cache_budget_bytes: u64,
    },
    VerifyMaterialization {
        #[arg(long)]
        output_dir: PathBuf,
    },
    KercEncode {
        #[arg(long)]
        code_vocab: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        kernel_offset: i32,
        #[arg(long)]
        pointer_offset: i32,
        #[arg(long, default_value_t = false)]
        include_tokens: bool,
    },
    KercBenchmark {
        #[arg(long)]
        code_vocab: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        kernel_offset: i32,
        #[arg(long)]
        pointer_offset: i32,
        #[arg(long, default_value_t = 3)]
        repetitions: usize,
    },
}

#[derive(Debug, Deserialize)]
struct InputRow {
    id: String,
    category: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct OutputRow {
    id: String,
    category: String,
    text_sha256: String,
    ids_sha256: String,
    ids: Vec<i32>,
    logical_tokens: Option<Vec<String>>,
    receipt: theseus_corpus::EncodingReceipt,
}

#[derive(Debug, Deserialize)]
struct KercInputRow {
    id: String,
    objective: String,
    target: String,
}

#[derive(Debug, Serialize)]
struct KercOutputRow {
    id: String,
    objective: String,
    target_sha256: String,
    ids_sha256: String,
    ids: Vec<i32>,
    tokens: Option<Vec<theseus_corpus::KercCodeToken>>,
    receipt: theseus_corpus::KercEncodingReceipt,
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run(args: Args) -> Result<(), String> {
    match args.command {
        Command::Encode {
            vocab,
            input,
            output,
            include_tokens,
        } => encode_jsonl(&vocab, &input, &output, include_tokens),
        Command::Benchmark {
            vocab,
            input,
            repetitions,
        } => benchmark_jsonl(&vocab, &input, repetitions),
        Command::ScannerBenchmark {
            input,
            repetitions,
            scalar,
        } => scanner_benchmark_jsonl(&input, repetitions, scalar),
        Command::Materialize {
            vocab,
            input,
            output_dir,
            target_offset,
            sequence_length,
            workers,
            chunk_rows,
        } => materialize_jsonl(
            &vocab,
            &input,
            &output_dir,
            target_offset,
            sequence_length,
            workers,
            chunk_rows,
        ),
        Command::CacheMaterialize {
            vocab,
            input,
            cache_root,
            target_offset,
            sequence_length,
            workers,
            chunk_rows,
            cache_budget_bytes,
        } => cache_materialize(
            &vocab,
            &input,
            &cache_root,
            target_offset,
            sequence_length,
            workers,
            chunk_rows,
            cache_budget_bytes,
        ),
        Command::VerifyMaterialization { output_dir } => verify_materialization(&output_dir),
        Command::KercEncode {
            code_vocab,
            input,
            output,
            kernel_offset,
            pointer_offset,
            include_tokens,
        } => kerc_encode_jsonl(
            &code_vocab,
            &input,
            &output,
            kernel_offset,
            pointer_offset,
            include_tokens,
        ),
        Command::KercBenchmark {
            code_vocab,
            input,
            kernel_offset,
            pointer_offset,
            repetitions,
        } => kerc_benchmark_jsonl(
            &code_vocab,
            &input,
            kernel_offset,
            pointer_offset,
            repetitions,
        ),
    }
}

fn kerc_encode_jsonl(
    code_vocab_path: &Path,
    input: &Path,
    output: &Path,
    kernel_offset: i32,
    pointer_offset: i32,
    include_tokens: bool,
) -> Result<(), String> {
    let kernel_vocab = ExactVocabulary::from_json_path_key(code_vocab_path, "kernel_vocab")?;
    let pointer_vocab = ExactVocabulary::from_json_path_key(code_vocab_path, "pointer_vocab")?;
    let source = BufReader::new(
        File::open(input).map_err(|error| format!("open {}: {error}", input.display()))?,
    );
    let temporary = output.with_extension(format!(
        "{}.tmp-{}",
        output
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("jsonl"),
        std::process::id()
    ));
    let mut sink = BufWriter::new(
        File::create(&temporary)
            .map_err(|error| format!("create {}: {error}", temporary.display()))?,
    );
    let result = (|| {
        for line in source.lines() {
            let row: KercInputRow = serde_json::from_str(
                &line.map_err(|error| format!("read {}: {error}", input.display()))?,
            )
            .map_err(|error| format!("parse KERC input row: {error}"))?;
            require_kerc_kernel_objective(&row.objective)?;
            let tokens = include_tokens
                .then(|| kerc_code_tokens(&row.target))
                .transpose()?;
            let (ids, receipt) = encode_kerc_global_target(
                &row.target,
                &kernel_vocab,
                &pointer_vocab,
                kernel_offset,
                pointer_offset,
            )?;
            if receipt.unknown_token_count != 0 || !receipt.exact_text_equal {
                return Err(format!("unrepresentable KERC row: {}", row.id));
            }
            let output_row = KercOutputRow {
                id: row.id,
                objective: row.objective,
                target_sha256: sha256(row.target.as_bytes()),
                ids_sha256: ids_sha256(&ids),
                ids,
                tokens,
                receipt,
            };
            serde_json::to_writer(&mut sink, &output_row)
                .map_err(|error| format!("write KERC output row: {error}"))?;
            sink.write_all(b"\n")
                .map_err(|error| format!("write KERC output delimiter: {error}"))?;
        }
        sink.flush()
            .map_err(|error| format!("flush {}: {error}", temporary.display()))
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    std::fs::rename(&temporary, output)
        .map_err(|error| format!("publish {}: {error}", output.display()))
}

fn kerc_benchmark_jsonl(
    code_vocab_path: &Path,
    input: &Path,
    kernel_offset: i32,
    pointer_offset: i32,
    repetitions: usize,
) -> Result<(), String> {
    if repetitions == 0 {
        return Err("repetitions must be positive".to_string());
    }
    let kernel_vocab = ExactVocabulary::from_json_path_key(code_vocab_path, "kernel_vocab")?;
    let pointer_vocab = ExactVocabulary::from_json_path_key(code_vocab_path, "pointer_vocab")?;
    let mut runs = Vec::new();
    for _ in 0..repetitions {
        let source = BufReader::new(
            File::open(input).map_err(|error| format!("open {}: {error}", input.display()))?,
        );
        let started = Instant::now();
        let mut rows = 0usize;
        let mut bytes = 0usize;
        let mut encoded_tokens = 0usize;
        let mut fallback_tokens = 0usize;
        let mut digest = Sha256::new();
        for line in source.lines() {
            let row: KercInputRow = serde_json::from_str(
                &line.map_err(|error| format!("read {}: {error}", input.display()))?,
            )
            .map_err(|error| format!("parse KERC input row: {error}"))?;
            require_kerc_kernel_objective(&row.objective)?;
            let (ids, receipt) = encode_kerc_global_target(
                &row.target,
                &kernel_vocab,
                &pointer_vocab,
                kernel_offset,
                pointer_offset,
            )?;
            if receipt.unknown_token_count != 0 || !receipt.exact_text_equal {
                return Err(format!("unrepresentable KERC row: {}", row.id));
            }
            rows += 1;
            bytes += row.target.len();
            encoded_tokens += ids.len();
            fallback_tokens += receipt.fallback_token_count;
            digest.update(row.id.as_bytes());
            for id in ids {
                digest.update(id.to_le_bytes());
            }
        }
        let seconds = started.elapsed().as_secs_f64();
        runs.push(serde_json::json!({
            "seconds": seconds,
            "rows": rows,
            "input_bytes": bytes,
            "encoded_tokens": encoded_tokens,
            "fallback_token_count": fallback_tokens,
            "mib_per_second": bytes as f64 / (1024.0 * 1024.0) / seconds.max(f64::MIN_POSITIVE),
            "encoded_tokens_per_second": encoded_tokens as f64 / seconds.max(f64::MIN_POSITIVE),
            "output_digest": format!("{:x}", digest.finalize()),
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "policy": "project_theseus_kerc_dual_space_encoding_rust_benchmark_v1",
            "implementation": "theseus-corpus",
            "kernel_offset": kernel_offset,
            "pointer_offset": pointer_offset,
            "runs": runs,
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "claim_boundary": "KERC tokenization and dual-space encoding station only",
        }))
        .map_err(|error| format!("serialize KERC benchmark: {error}"))?
    );
    Ok(())
}

fn require_kerc_kernel_objective(objective: &str) -> Result<(), String> {
    if KERC_KERNEL_OBJECTIVES.contains(&objective) {
        Ok(())
    } else {
        Err(format!("unsupported KERC kernel objective: {objective}"))
    }
}

fn materialize_jsonl(
    vocab_path: &Path,
    input: &Path,
    output_dir: &Path,
    target_offset: i32,
    sequence_length: usize,
    workers: usize,
    chunk_rows: usize,
) -> Result<(), String> {
    let summary = publish_materialization(
        vocab_path,
        input,
        output_dir,
        target_offset,
        sequence_length,
        workers,
        chunk_rows,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&summary)
            .map_err(|error| format!("serialize materialization receipt: {error}"))?
    );
    Ok(())
}

fn publish_materialization(
    vocab_path: &Path,
    input: &Path,
    output_dir: &Path,
    target_offset: i32,
    sequence_length: usize,
    workers: usize,
    chunk_rows: usize,
) -> Result<serde_json::Value, String> {
    if sequence_length == 0 {
        return Err("sequence_length must be positive".to_string());
    }
    if workers == 0 || chunk_rows == 0 {
        return Err("workers and chunk_rows must be positive".to_string());
    }
    if output_dir.exists() {
        return Err(format!(
            "output directory already exists: {}",
            output_dir.display()
        ));
    }
    let parent = output_dir
        .parent()
        .ok_or_else(|| "output directory requires a parent".to_string())?;
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("create {}: {error}", parent.display()))?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        output_dir
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("theseus-corpus"),
        std::process::id()
    ));
    if temporary.exists() {
        std::fs::remove_dir_all(&temporary)
            .map_err(|error| format!("remove stale {}: {error}", temporary.display()))?;
    }
    std::fs::create_dir(&temporary)
        .map_err(|error| format!("create {}: {error}", temporary.display()))?;
    let result = materialize_jsonl_inner(
        vocab_path,
        input,
        &temporary,
        target_offset,
        sequence_length,
        workers,
        chunk_rows,
    );
    let summary = match result {
        Ok(summary) => summary,
        Err(error) => {
            let _ = std::fs::remove_dir_all(&temporary);
            return Err(error);
        }
    };
    let manifest_path = temporary.join("materialization_manifest_v1.json");
    let manifest_bytes = serde_json::to_vec_pretty(&summary)
        .map_err(|error| format!("serialize materialization manifest: {error}"))?;
    std::fs::write(&manifest_path, manifest_bytes)
        .map_err(|error| format!("write {}: {error}", manifest_path.display()))?;
    std::fs::rename(&temporary, output_dir).map_err(|error| {
        let _ = std::fs::remove_dir_all(&temporary);
        format!("publish {}: {error}", output_dir.display())
    })?;
    Ok(summary)
}

fn materialize_jsonl_inner(
    vocab_path: &Path,
    input: &Path,
    output_dir: &Path,
    target_offset: i32,
    sequence_length: usize,
    workers: usize,
    chunk_rows: usize,
) -> Result<serde_json::Value, String> {
    let vocab = ExactVocabulary::from_json_path(vocab_path)?;
    let source = open_text_reader(input)?;
    let pool = (workers > 1)
        .then(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .thread_name(|index| format!("theseus-corpus-{index}"))
                .build()
                .map_err(|error| format!("build bounded corpus worker pool: {error}"))
        })
        .transpose()?;
    let input_path = output_dir.join("canonical_pretrain_inputs_v1.i32");
    let label_path = output_dir.join("canonical_pretrain_labels_v1.i32");
    let mask_path = output_dir.join("canonical_pretrain_mask_v1.u8");
    let mut input_sink = BufWriter::new(
        File::create(&input_path)
            .map_err(|error| format!("create {}: {error}", input_path.display()))?,
    );
    let mut label_sink = BufWriter::new(
        File::create(&label_path)
            .map_err(|error| format!("create {}: {error}", label_path.display()))?,
    );
    let mut mask_sink = BufWriter::new(
        File::create(&mask_path)
            .map_err(|error| format!("create {}: {error}", mask_path.display()))?,
    );
    let mut input_row = vec![0i32; sequence_length];
    let mut label_row = vec![0i32; sequence_length];
    let mut mask_row = vec![0u8; sequence_length];
    let mut document_count = 0usize;
    let mut window_count = 0usize;
    let mut materialized_positions = 0usize;
    let mut fallback_tokens = 0usize;
    let mut fallback_bytes = 0usize;
    let mut selected = Sha256::new();
    let started = Instant::now();
    let mut lines = source.lines();
    loop {
        let mut rows = Vec::with_capacity(chunk_rows);
        for _ in 0..chunk_rows {
            let Some(line) = lines.next() else { break };
            let line = line.map_err(|error| format!("read {}: {error}", input.display()))?;
            rows.push(
                serde_json::from_str::<InputRow>(&line)
                    .map_err(|error| format!("parse input row: {error}"))?,
            );
        }
        if rows.is_empty() {
            break;
        }
        let encode = || {
            rows.par_iter()
                .map(|row| prepare_document(&vocab, row, target_offset))
                .collect::<Result<Vec<_>, _>>()
        };
        let encoded = if let Some(pool) = &pool {
            pool.install(encode)?
        } else {
            rows.iter()
                .map(|row| prepare_document(&vocab, row, target_offset))
                .collect::<Result<Vec<_>, _>>()?
        };
        for document in encoded {
            fallback_tokens += document.fallback_token_count;
            fallback_bytes += document.fallback_byte_count;
            document_count += 1;
            selected.update(document.category.as_bytes());
            selected.update(b":");
            selected.update(document.id.as_bytes());
            selected.update(b"\n");
            for start in (0..document.ids.len().saturating_sub(1)).step_by(sequence_length) {
                let ids = &document.ids;
                let width = sequence_length.min(ids.len() - start - 1);
                if width == 0 {
                    continue;
                }
                input_row.fill(0);
                label_row.fill(0);
                mask_row.fill(0);
                input_row[..width].copy_from_slice(&ids[start..start + width]);
                label_row[..width].copy_from_slice(&ids[start + 1..start + width + 1]);
                mask_row[..width].fill(1);
                write_i32_row(&mut input_sink, &input_row)?;
                write_i32_row(&mut label_sink, &label_row)?;
                mask_sink
                    .write_all(&mask_row)
                    .map_err(|error| format!("write mask row: {error}"))?;
                materialized_positions += width;
                window_count += 1;
            }
        }
    }
    input_sink
        .flush()
        .map_err(|error| format!("flush inputs: {error}"))?;
    label_sink
        .flush()
        .map_err(|error| format!("flush labels: {error}"))?;
    mask_sink
        .flush()
        .map_err(|error| format!("flush mask: {error}"))?;
    let seconds = started.elapsed().as_secs_f64();
    let artifacts = [
        ("inputs", &input_path),
        ("labels", &label_path),
        ("mask", &mask_path),
    ]
    .into_iter()
    .map(|(key, path)| {
        Ok((
            key.to_string(),
            serde_json::json!({
                "path": path.file_name().and_then(|value| value.to_str()),
                "bytes": path.metadata().map_err(|error| format!("stat {}: {error}", path.display()))?.len(),
                "sha256": file_sha256(path)?,
            }),
        ))
    })
    .collect::<Result<serde_json::Map<String, serde_json::Value>, String>>()?;
    let vocabulary_sha256 = file_sha256(vocab_path)?;
    let source_sha256 = file_sha256(input)?;
    let materialization_identity_sha256 = materialization_identity(
        &vocabulary_sha256,
        &source_sha256,
        target_offset,
        sequence_length,
    );
    Ok(serde_json::json!({
        "policy": "project_theseus_exact_corpus_to_tensor_rust_v1",
        "implementation": "theseus-corpus",
        "document_count": document_count,
        "window_count": window_count,
        "materialized_positions": materialized_positions,
        "sequence_length": sequence_length,
        "target_offset": target_offset,
        "input_codec": input_codec(input),
        "worker_count": workers,
        "chunk_rows": chunk_rows,
        "bounded_parallel_reorder_buffer_rows": if workers > 1 { chunk_rows } else { 0 },
        "deterministic_row_order": true,
        "vocabulary_sha256": vocabulary_sha256,
        "source_sha256": source_sha256,
        "materialization_identity_sha256": materialization_identity_sha256,
        "fallback_token_count": fallback_tokens,
        "fallback_byte_count": fallback_bytes,
        "selected_document_digest": format!("{:x}", selected.finalize()),
        "wall_seconds": seconds,
        "positions_per_second": materialized_positions as f64 / seconds.max(f64::MIN_POSITIVE),
        "artifacts": artifacts,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "claim_boundary": "exact corpus-to-tensor mechanics only; not training or capability evidence",
    }))
}

#[allow(clippy::too_many_arguments)]
fn cache_materialize(
    vocab_path: &Path,
    input: &Path,
    cache_root: &Path,
    target_offset: i32,
    sequence_length: usize,
    workers: usize,
    chunk_rows: usize,
    cache_budget_bytes: u64,
) -> Result<(), String> {
    if cache_budget_bytes == 0 {
        return Err("cache_budget_bytes must be positive".to_string());
    }
    std::fs::create_dir_all(cache_root)
        .map_err(|error| format!("create {}: {error}", cache_root.display()))?;
    let vocabulary_sha256 = file_sha256(vocab_path)?;
    let source_sha256 = file_sha256(input)?;
    let identity = materialization_identity(
        &vocabulary_sha256,
        &source_sha256,
        target_offset,
        sequence_length,
    );
    let entry = cache_root.join(&identity);
    let cache_hit = entry.exists();
    let manifest = if cache_hit {
        verify_materialization_value(&entry)?;
        read_manifest(&entry)?
    } else {
        publish_materialization(
            vocab_path,
            input,
            &entry,
            target_offset,
            sequence_length,
            workers,
            chunk_rows,
        )?
    };
    for (field, expected) in [
        ("vocabulary_sha256", vocabulary_sha256.as_str()),
        ("source_sha256", source_sha256.as_str()),
        ("materialization_identity_sha256", identity.as_str()),
    ] {
        if manifest.get(field).and_then(serde_json::Value::as_str) != Some(expected) {
            if !cache_hit {
                let _ = std::fs::remove_dir_all(&entry);
            }
            return Err(format!("cache identity binding mismatch: {field}"));
        }
    }
    if manifest
        .get("target_offset")
        .and_then(serde_json::Value::as_i64)
        != Some(i64::from(target_offset))
        || manifest
            .get("sequence_length")
            .and_then(serde_json::Value::as_u64)
            != Some(sequence_length as u64)
    {
        if !cache_hit {
            let _ = std::fs::remove_dir_all(&entry);
        }
        return Err("cache tensor contract mismatch".to_string());
    }
    let access_root = cache_root.join(".access");
    std::fs::create_dir_all(&access_root)
        .map_err(|error| format!("create {}: {error}", access_root.display()))?;
    let access_path = access_root.join(&identity);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock before Unix epoch: {error}"))?
        .as_nanos();
    std::fs::write(&access_path, now.to_string())
        .map_err(|error| format!("write {}: {error}", access_path.display()))?;
    let cache = match enforce_cache_budget(cache_root, &identity, cache_budget_bytes) {
        Ok(value) => value,
        Err(error) => {
            if !cache_hit {
                let _ = std::fs::remove_dir_all(&entry);
                let _ = std::fs::remove_file(&access_path);
            }
            return Err(error);
        }
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "policy": "project_theseus_content_addressed_corpus_cache_v1",
            "state": "GREEN",
            "cache_hit": cache_hit,
            "cache_root": cache_root,
            "entry_path": entry,
            "materialization_identity_sha256": identity,
            "manifest_sha256": file_sha256(&entry.join("materialization_manifest_v1.json"))?,
            "cache": cache,
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "claim_boundary": "verified immutable corpus materialization reuse only",
        }))
        .map_err(|error| format!("serialize cache receipt: {error}"))?
    );
    Ok(())
}

fn materialization_identity(
    vocabulary_sha256: &str,
    source_sha256: &str,
    target_offset: i32,
    sequence_length: usize,
) -> String {
    sha256(
        format!(
            "project_theseus_exact_corpus_to_tensor_rust_v1\n{vocabulary_sha256}\n{source_sha256}\n{target_offset}\n{sequence_length}\n"
        )
        .as_bytes(),
    )
}

fn read_manifest(output_dir: &Path) -> Result<serde_json::Value, String> {
    let path = output_dir.join("materialization_manifest_v1.json");
    serde_json::from_slice(
        &std::fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("parse {}: {error}", path.display()))
}

fn directory_size(path: &Path) -> Result<u64, String> {
    let mut total = 0_u64;
    for entry in std::fs::read_dir(path)
        .map_err(|error| format!("read directory {}: {error}", path.display()))?
    {
        let entry = entry.map_err(|error| format!("read directory entry: {error}"))?;
        let metadata = entry
            .metadata()
            .map_err(|error| format!("stat {}: {error}", entry.path().display()))?;
        total = total
            .checked_add(if metadata.is_dir() {
                directory_size(&entry.path())?
            } else {
                metadata.len()
            })
            .ok_or_else(|| "cache byte count overflow".to_string())?;
    }
    Ok(total)
}

fn enforce_cache_budget(
    cache_root: &Path,
    protected_identity: &str,
    budget_bytes: u64,
) -> Result<serde_json::Value, String> {
    let access_root = cache_root.join(".access");
    let mut entries = Vec::new();
    for row in std::fs::read_dir(cache_root)
        .map_err(|error| format!("read directory {}: {error}", cache_root.display()))?
    {
        let row = row.map_err(|error| format!("read cache entry: {error}"))?;
        if !row
            .file_type()
            .map_err(|error| format!("read cache entry type: {error}"))?
            .is_dir()
            || row.file_name() == ".access"
        {
            continue;
        }
        let identity = row.file_name().to_string_lossy().into_owned();
        let bytes = directory_size(&row.path())?;
        let access = std::fs::read_to_string(access_root.join(&identity))
            .ok()
            .and_then(|value| value.parse::<u128>().ok())
            .unwrap_or(0);
        entries.push((access, identity, bytes, row.path()));
    }
    let mut total = entries.iter().map(|row| row.2).sum::<u64>();
    let protected_bytes = entries
        .iter()
        .find(|row| row.1 == protected_identity)
        .map(|row| row.2)
        .unwrap_or(0);
    if protected_bytes > budget_bytes {
        return Err(format!(
            "current materialization exceeds cache budget: {protected_bytes} > {budget_bytes}"
        ));
    }
    entries.sort_by_key(|row| row.0);
    let mut evicted = Vec::new();
    for (_, identity, bytes, path) in entries {
        if total <= budget_bytes {
            break;
        }
        if identity == protected_identity {
            continue;
        }
        std::fs::remove_dir_all(&path)
            .map_err(|error| format!("evict {}: {error}", path.display()))?;
        let _ = std::fs::remove_file(access_root.join(&identity));
        total -= bytes;
        evicted.push(identity);
    }
    Ok(serde_json::json!({
        "budget_bytes": budget_bytes,
        "resident_bytes": total,
        "evicted_entry_count": evicted.len(),
        "evicted_identities": evicted,
        "protected_current_entry": protected_identity,
    }))
}

#[derive(Debug)]
struct PreparedDocument {
    id: String,
    category: String,
    ids: Vec<i32>,
    fallback_token_count: usize,
    fallback_byte_count: usize,
}

fn prepare_document(
    vocab: &ExactVocabulary,
    row: &InputRow,
    target_offset: i32,
) -> Result<PreparedDocument, String> {
    let encoded = vocab.encode_document(&row.text, &row.category)?;
    if encoded.receipt.unknown_token_count != 0 || !encoded.receipt.exact_text_equal {
        return Err(format!("unrepresentable document: {}", row.id));
    }
    let ids = encoded
        .ids
        .into_iter()
        .map(|value| {
            value
                .checked_add(target_offset)
                .ok_or_else(|| format!("target id overflow: {}", row.id))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PreparedDocument {
        id: row.id.clone(),
        category: row.category.clone(),
        ids,
        fallback_token_count: encoded.receipt.fallback_token_count,
        fallback_byte_count: encoded.receipt.fallback_byte_count,
    })
}

fn input_codec(path: &Path) -> &'static str {
    match path.extension().and_then(|value| value.to_str()) {
        Some("gz") => "gzip",
        Some("zst" | "zstd") => "zstd",
        _ => "plain",
    }
}

fn open_text_reader(path: &Path) -> Result<Box<dyn BufRead>, String> {
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    match input_codec(path) {
        "gzip" => Ok(Box::new(BufReader::new(GzDecoder::new(file)))),
        "zstd" => Ok(Box::new(BufReader::new(
            zstd::stream::read::Decoder::new(file)
                .map_err(|error| format!("open zstd stream {}: {error}", path.display()))?,
        ))),
        _ => Ok(Box::new(BufReader::new(file))),
    }
}

fn verify_materialization(output_dir: &Path) -> Result<(), String> {
    let receipt = verify_materialization_value(output_dir)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt)
            .map_err(|error| format!("serialize verification receipt: {error}"))?
    );
    Ok(())
}

fn verify_materialization_value(output_dir: &Path) -> Result<serde_json::Value, String> {
    let manifest_path = output_dir.join("materialization_manifest_v1.json");
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&manifest_path)
            .map_err(|error| format!("read {}: {error}", manifest_path.display()))?,
    )
    .map_err(|error| format!("parse {}: {error}", manifest_path.display()))?;
    if manifest.get("policy").and_then(serde_json::Value::as_str)
        != Some("project_theseus_exact_corpus_to_tensor_rust_v1")
    {
        return Err("materialization manifest policy mismatch".to_string());
    }
    let artifacts = manifest
        .get("artifacts")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| "materialization manifest artifacts missing".to_string())?;
    let mut verified = serde_json::Map::new();
    for key in ["inputs", "labels", "mask"] {
        let artifact = artifacts
            .get(key)
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| format!("materialization artifact missing: {key}"))?;
        let relative = artifact
            .get("path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("materialization artifact path missing: {key}"))?;
        let path = output_dir.join(relative);
        let expected_bytes = artifact
            .get("bytes")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| format!("materialization artifact size missing: {key}"))?;
        let expected_sha256 = artifact
            .get("sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("materialization artifact digest missing: {key}"))?;
        let observed_bytes = path
            .metadata()
            .map_err(|error| format!("stat {}: {error}", path.display()))?
            .len();
        let observed_sha256 = file_sha256(&path)?;
        if observed_bytes != expected_bytes || observed_sha256 != expected_sha256 {
            return Err(format!("materialization artifact integrity fault: {key}"));
        }
        verified.insert(
            key.to_string(),
            serde_json::json!({
                "bytes": observed_bytes,
                "sha256": observed_sha256,
            }),
        );
    }
    Ok(serde_json::json!({
        "policy": "project_theseus_corpus_materialization_verify_v1",
        "state": "GREEN",
        "materialization_identity_sha256": manifest.get("materialization_identity_sha256"),
        "verified_artifacts": verified,
        "fallback_return_count": 0,
    }))
}

fn write_i32_row(sink: &mut BufWriter<File>, row: &[i32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(row));
    for value in row {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    sink.write_all(&bytes)
        .map_err(|error| format!("write i32 row: {error}"))
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let mut source = BufReader::new(
        File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?,
    );
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = std::io::Read::read(&mut source, &mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn encode_jsonl(
    vocab_path: &Path,
    input: &Path,
    output: &Path,
    include_tokens: bool,
) -> Result<(), String> {
    let vocab = ExactVocabulary::from_json_path(vocab_path)?;
    let source = BufReader::new(
        File::open(input).map_err(|error| format!("open {}: {error}", input.display()))?,
    );
    let temporary = output.with_extension(format!(
        "{}.tmp-{}",
        output
            .extension()
            .and_then(|x| x.to_str())
            .unwrap_or("jsonl"),
        std::process::id()
    ));
    let mut sink = BufWriter::new(
        File::create(&temporary)
            .map_err(|error| format!("create {}: {error}", temporary.display()))?,
    );
    let result = (|| {
        for line in source.lines() {
            let line = line.map_err(|error| format!("read {}: {error}", input.display()))?;
            let row: InputRow =
                serde_json::from_str(&line).map_err(|error| format!("parse input row: {error}"))?;
            let encoded = vocab.encode_document(&row.text, &row.category)?;
            let output_row = OutputRow {
                id: row.id,
                category: row.category,
                text_sha256: sha256(row.text.as_bytes()),
                ids_sha256: ids_sha256(&encoded.ids),
                ids: encoded.ids,
                logical_tokens: include_tokens.then_some(encoded.logical_tokens),
                receipt: encoded.receipt,
            };
            serde_json::to_writer(&mut sink, &output_row)
                .map_err(|error| format!("write output row: {error}"))?;
            sink.write_all(b"\n")
                .map_err(|error| format!("write output delimiter: {error}"))?;
        }
        sink.flush()
            .map_err(|error| format!("flush {}: {error}", temporary.display()))
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    std::fs::rename(&temporary, output)
        .map_err(|error| format!("publish {}: {error}", output.display()))
}

fn benchmark_jsonl(vocab_path: &Path, input: &Path, repetitions: usize) -> Result<(), String> {
    if repetitions == 0 {
        return Err("repetitions must be positive".to_string());
    }
    let vocab = ExactVocabulary::from_json_path(vocab_path)?;
    let mut runs = Vec::new();
    for _ in 0..repetitions {
        let started = Instant::now();
        let source = BufReader::new(
            File::open(input).map_err(|error| format!("open {}: {error}", input.display()))?,
        );
        let mut rows = 0usize;
        let mut bytes = 0usize;
        let mut logical_tokens = 0usize;
        let mut encoded_tokens = 0usize;
        let mut unknown = 0usize;
        let mut digest = Sha256::new();
        for line in source.lines() {
            let line = line.map_err(|error| format!("read {}: {error}", input.display()))?;
            let row: InputRow =
                serde_json::from_str(&line).map_err(|error| format!("parse input row: {error}"))?;
            let encoded = vocab.encode_document(&row.text, &row.category)?;
            rows += 1;
            bytes += row.text.len();
            logical_tokens += encoded.logical_tokens.len();
            encoded_tokens += encoded.ids.len();
            unknown += encoded.receipt.unknown_token_count;
            digest.update(row.id.as_bytes());
            for id in encoded.ids {
                digest.update(id.to_le_bytes());
            }
        }
        let seconds = started.elapsed().as_secs_f64();
        runs.push(serde_json::json!({
            "seconds": seconds,
            "rows": rows,
            "input_bytes": bytes,
            "logical_tokens": logical_tokens,
            "encoded_tokens": encoded_tokens,
            "unknown_token_count": unknown,
            "mib_per_second": bytes as f64 / (1024.0 * 1024.0) / seconds.max(f64::MIN_POSITIVE),
            "encoded_tokens_per_second": encoded_tokens as f64 / seconds.max(f64::MIN_POSITIVE),
            "output_digest": format!("{:x}", digest.finalize()),
        }));
    }
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "policy": "project_theseus_rust_exact_tokenizer_benchmark_v1",
        "implementation": "theseus-corpus",
        "input": input,
        "vocabulary": vocab_path,
        "runs": runs,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "claim_boundary": "tokenizer station evidence only; not end-to-end training or capability evidence",
    })).map_err(|error| format!("serialize benchmark: {error}"))?);
    Ok(())
}

fn scanner_benchmark_jsonl(input: &Path, repetitions: usize, scalar: bool) -> Result<(), String> {
    if repetitions == 0 {
        return Err("repetitions must be positive".to_string());
    }
    let mut runs = Vec::new();
    for _ in 0..repetitions {
        let started = Instant::now();
        let source = open_text_reader(input)?;
        let mut rows = 0usize;
        let mut bytes = 0usize;
        let mut logical_tokens = 0usize;
        let mut digest = Sha256::new();
        for line in source.lines() {
            let line = line.map_err(|error| format!("read {}: {error}", input.display()))?;
            let row: InputRow =
                serde_json::from_str(&line).map_err(|error| format!("parse input row: {error}"))?;
            let tokens = if scalar {
                exact_text_tokens_scalar(&row.text)
            } else {
                exact_text_tokens(&row.text)
            };
            rows += 1;
            bytes += row.text.len();
            logical_tokens += tokens.len();
            digest.update(row.id.as_bytes());
            for token in tokens {
                digest.update((token.len() as u64).to_le_bytes());
                digest.update(token.as_bytes());
            }
        }
        let seconds = started.elapsed().as_secs_f64();
        runs.push(serde_json::json!({
            "seconds": seconds,
            "rows": rows,
            "input_bytes": bytes,
            "logical_tokens": logical_tokens,
            "mib_per_second": bytes as f64 / (1024.0 * 1024.0) / seconds.max(f64::MIN_POSITIVE),
            "output_digest": format!("{:x}", digest.finalize()),
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "policy": "project_theseus_exact_scanner_benchmark_v1",
            "implementation": if scalar { "scalar_reference" } else { "aarch64_neon_bounded_ascii_runs" },
            "input": input,
            "input_codec": input_codec(input),
            "runs": runs,
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "claim_boundary": "exact scanner station evidence only; not corpus-to-tensor, training, or capability evidence",
        }))
        .map_err(|error| format!("serialize scanner benchmark: {error}"))?
    );
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}
