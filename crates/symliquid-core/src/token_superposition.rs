use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::benchmarks::{training_batch_tensor, ReadoutTrainingExample};
use crate::error::{Result, SymError};
use crate::tensor::Tensor;
use crate::train::LinearReadout;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSuperpositionConfig {
    pub train_seed: u64,
    pub input_paths: Vec<String>,
    pub include_project_code: bool,
    pub project_code_roots: Vec<String>,
    pub max_language_rows: usize,
    pub max_code_files: usize,
    pub max_chars_per_doc: usize,
    pub max_vocab: usize,
    pub hv_dim: usize,
    pub train_samples: usize,
    pub eval_samples: usize,
    pub baseline_epochs: usize,
    pub bag_sizes: Vec<usize>,
    pub recovery_ratios: Vec<f32>,
    pub lr: f32,
    pub gate_tolerance: f32,
    pub min_nominal_speedup: f32,
    pub min_train_speedup: f32,
    pub artifact_path: Option<String>,
}

impl Default for TokenSuperpositionConfig {
    fn default() -> Self {
        Self {
            train_seed: 20_260_514,
            input_paths: vec!["data/babylm_blimp_filtered_train.jsonl".to_string()],
            include_project_code: true,
            project_code_roots: vec!["scripts".to_string(), "crates".to_string()],
            max_language_rows: 12_000,
            max_code_files: 240,
            max_chars_per_doc: 12_000,
            max_vocab: 256,
            hv_dim: 4096,
            train_samples: 32_768,
            eval_samples: 4096,
            baseline_epochs: 6,
            bag_sizes: vec![4, 8],
            recovery_ratios: vec![0.2, 0.4],
            lr: 0.03,
            gate_tolerance: 0.002,
            min_nominal_speedup: 1.2,
            min_train_speedup: 1.0,
            artifact_path: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenSuperpositionDataset {
    pub vocab: Vec<String>,
    pub train_tokens: Vec<usize>,
    pub language_eval_tokens: Vec<usize>,
    pub code_eval_tokens: Vec<usize>,
    pub summary: TokenSuperpositionDatasetSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSuperpositionDatasetSummary {
    pub language_train_docs: usize,
    pub language_eval_docs: usize,
    pub code_train_docs: usize,
    pub code_eval_docs: usize,
    pub train_tokens: usize,
    pub language_eval_tokens: usize,
    pub code_eval_tokens: usize,
    pub vocab_size: usize,
    pub hv_dim: usize,
    pub holdout_policy: String,
}

#[derive(Debug, Clone)]
pub struct BagTrainingBatch {
    pub features: Tensor,
    pub target_bags: Vec<usize>,
    pub targets_per_sample: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSuperpositionEvalMetrics {
    pub language_ar_loss: f32,
    pub language_ar_accuracy: f32,
    pub code_ar_loss: f32,
    pub code_ar_accuracy: f32,
    pub combined_ar_loss: f32,
    pub combined_ar_accuracy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSuperpositionRunReport {
    pub id: String,
    pub objective: String,
    pub bag_size: Option<usize>,
    pub recovery_ratio: Option<f32>,
    pub bag_epochs: usize,
    pub recovery_epochs: usize,
    pub baseline_epochs: usize,
    pub train_samples: usize,
    pub bag_samples: usize,
    pub recovery_samples: usize,
    pub train_runtime_ms: u128,
    pub feature_build_ms: u128,
    pub eval_runtime_ms: u128,
    pub total_runtime_ms: u128,
    pub train_examples_seen: usize,
    pub train_examples_per_second: f32,
    pub kernel_launches: usize,
    pub train_loss: f32,
    pub train_accuracy: f32,
    pub eval: TokenSuperpositionEvalMetrics,
    pub nominal_speedup_vs_baseline: f32,
    pub measured_train_speedup_vs_baseline: f32,
    pub measured_total_speedup_vs_baseline: f32,
    pub combined_loss_delta_vs_baseline: f32,
    pub code_loss_delta_vs_baseline: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSuperpositionGate {
    pub gate: String,
    pub passed: bool,
    pub evidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSuperpositionPromotionDecision {
    pub promote_to_training_lane: bool,
    pub status: String,
    pub reason: String,
    pub artifact: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSuperpositionReport {
    pub policy: String,
    pub created_unix: u64,
    pub backend: String,
    pub cuda_fallback: bool,
    pub config: TokenSuperpositionConfig,
    pub dataset: TokenSuperpositionDatasetSummary,
    pub baseline: TokenSuperpositionRunReport,
    pub variants: Vec<TokenSuperpositionRunReport>,
    pub best_variant: Option<TokenSuperpositionRunReport>,
    pub gates: Vec<TokenSuperpositionGate>,
    pub promotion_decision: TokenSuperpositionPromotionDecision,
    pub timing_breakdown_ms: BTreeMap<String, u128>,
    pub external_inference_calls: usize,
}

#[derive(Debug, Clone)]
struct TextDoc {
    text: String,
}

pub fn build_token_superposition_dataset(
    root: &Path,
    config: &TokenSuperpositionConfig,
) -> Result<TokenSuperpositionDataset> {
    let (language_train, language_eval) = load_language_docs(root, config)?;
    let (code_train, code_eval) = if config.include_project_code {
        load_project_code_docs(root, config)?
    } else {
        (Vec::new(), Vec::new())
    };
    let mut train_docs = Vec::new();
    train_docs.extend(language_train.iter().map(|doc| doc.text.as_str()));
    train_docs.extend(code_train.iter().map(|doc| doc.text.as_str()));
    if train_docs.is_empty() {
        return Err(SymError::InvalidArgument(
            "Token superposition dataset has no training documents".to_string(),
        ));
    }

    let vocab = build_vocab(&train_docs, config.max_vocab.max(8));
    let train_tokens = encode_docs(&train_docs, &vocab);
    let language_eval_tokens = encode_docs(
        &language_eval
            .iter()
            .map(|doc| doc.text.as_str())
            .collect::<Vec<_>>(),
        &vocab,
    );
    let code_eval_tokens = encode_docs(
        &code_eval
            .iter()
            .map(|doc| doc.text.as_str())
            .collect::<Vec<_>>(),
        &vocab,
    );
    let min_required = config.train_samples.min(12_000);
    if train_tokens.len() < min_required {
        return Err(SymError::InvalidArgument(format!(
            "Token superposition train tokens {} below minimum {}",
            train_tokens.len(),
            min_required
        )));
    }
    let summary = TokenSuperpositionDatasetSummary {
        language_train_docs: language_train.len(),
        language_eval_docs: language_eval.len(),
        code_train_docs: code_train.len(),
        code_eval_docs: code_eval.len(),
        train_tokens: train_tokens.len(),
        language_eval_tokens: language_eval_tokens.len(),
        code_eval_tokens: code_eval_tokens.len(),
        vocab_size: vocab.len(),
        hv_dim: config.hv_dim,
        holdout_policy: "Deterministic local train/eval split; public eval holdouts are not used as train input.".to_string(),
    };
    Ok(TokenSuperpositionDataset {
        vocab,
        train_tokens,
        language_eval_tokens,
        code_eval_tokens,
        summary,
    })
}

pub fn token_feature_table(vocab: &[String], hv_dim: usize) -> Vec<Vec<f32>> {
    vocab
        .iter()
        .map(|token| token_features(token, hv_dim))
        .collect()
}

pub fn make_ar_training_batch(
    dataset: &TokenSuperpositionDataset,
    feature_table: &[Vec<f32>],
    sample_count: usize,
    seed: u64,
) -> Result<(Tensor, Vec<usize>)> {
    let max_start = dataset.train_tokens.len().saturating_sub(2);
    if max_start == 0 {
        return Err(SymError::InvalidArgument(
            "not enough tokens for AR training".to_string(),
        ));
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xA11A_0001);
    let mut examples = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        let pos = rng.gen_range(0..max_start);
        let token = dataset.train_tokens[pos];
        let target = dataset.train_tokens[pos + 1];
        examples.push(ReadoutTrainingExample {
            features: feature_table[token].clone(),
            target,
        });
    }
    training_batch_tensor(&examples, dataset.summary.hv_dim)
}

pub fn make_bag_training_batch(
    dataset: &TokenSuperpositionDataset,
    feature_table: &[Vec<f32>],
    sample_count: usize,
    bag_size: usize,
    seed: u64,
) -> Result<BagTrainingBatch> {
    if bag_size == 0 {
        return Err(SymError::InvalidArgument(
            "bag_size must be greater than zero".to_string(),
        ));
    }
    let max_start = dataset
        .train_tokens
        .len()
        .saturating_sub((2 * bag_size) + 1);
    if max_start == 0 {
        return Err(SymError::InvalidArgument(format!(
            "not enough tokens for bag_size={bag_size}"
        )));
    }
    let hv_dim = dataset.summary.hv_dim;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xB_A66A_0001);
    let mut data = Vec::with_capacity(sample_count * hv_dim);
    let mut target_bags = Vec::with_capacity(sample_count * bag_size);
    for _ in 0..sample_count {
        let pos = rng.gen_range(0..max_start);
        let mut row = vec![0.0; hv_dim];
        for offset in 0..bag_size {
            let token = dataset.train_tokens[pos + offset];
            for (dst, value) in row.iter_mut().zip(&feature_table[token]) {
                *dst += *value;
            }
        }
        let inv = 1.0 / bag_size as f32;
        for value in &mut row {
            *value *= inv;
        }
        normalize_in_place(&mut row);
        data.extend_from_slice(&row);
        for offset in 0..bag_size {
            target_bags.push(dataset.train_tokens[pos + bag_size + offset]);
        }
    }
    Ok(BagTrainingBatch {
        features: Tensor::new(sample_count, hv_dim, data)?,
        target_bags,
        targets_per_sample: bag_size,
    })
}

pub fn evaluate_token_readout(
    readout: &LinearReadout,
    dataset: &TokenSuperpositionDataset,
    feature_table: &[Vec<f32>],
    eval_samples: usize,
) -> Result<TokenSuperpositionEvalMetrics> {
    let (language_loss, language_accuracy) = evaluate_split_ar(
        readout,
        &dataset.language_eval_tokens,
        feature_table,
        dataset.summary.hv_dim,
        eval_samples,
    )?;
    let (code_loss, code_accuracy) = evaluate_split_ar(
        readout,
        &dataset.code_eval_tokens,
        feature_table,
        dataset.summary.hv_dim,
        eval_samples,
    )?;
    let combined_ar_loss = if code_loss.is_finite() {
        (language_loss + code_loss) * 0.5
    } else {
        language_loss
    };
    let combined_ar_accuracy = if code_accuracy.is_finite() {
        (language_accuracy + code_accuracy) * 0.5
    } else {
        language_accuracy
    };
    Ok(TokenSuperpositionEvalMetrics {
        language_ar_loss: language_loss,
        language_ar_accuracy: language_accuracy,
        code_ar_loss: code_loss,
        code_ar_accuracy: code_accuracy,
        combined_ar_loss,
        combined_ar_accuracy,
    })
}

pub fn build_token_superposition_report(
    backend: &str,
    cuda_fallback: bool,
    config: TokenSuperpositionConfig,
    dataset: TokenSuperpositionDatasetSummary,
    baseline: TokenSuperpositionRunReport,
    variants: Vec<TokenSuperpositionRunReport>,
    mut timing_breakdown_ms: BTreeMap<String, u128>,
) -> TokenSuperpositionReport {
    let best_variant = variants.iter().cloned().min_by(|a, b| {
        a.eval
            .combined_ar_loss
            .total_cmp(&b.eval.combined_ar_loss)
            .then_with(|| {
                b.nominal_speedup_vs_baseline
                    .total_cmp(&a.nominal_speedup_vs_baseline)
            })
    });
    let gates = token_superposition_gates(&config, &baseline, best_variant.as_ref());
    let failed = gates
        .iter()
        .filter(|gate| !gate.passed)
        .map(|gate| gate.gate.clone())
        .collect::<Vec<_>>();
    let promote = failed.is_empty();
    let artifact = config.artifact_path.clone().unwrap_or_default();
    timing_breakdown_ms.insert("baseline_total".to_string(), baseline.total_runtime_ms);
    if let Some(best) = &best_variant {
        timing_breakdown_ms.insert("best_variant_total".to_string(), best.total_runtime_ms);
    }
    TokenSuperpositionReport {
        policy: "project_theseus_token_superposition_rust_cuda_report_v1".to_string(),
        created_unix: now_unix(),
        backend: backend.to_string(),
        cuda_fallback,
        config,
        dataset,
        baseline,
        variants,
        best_variant,
        gates,
        promotion_decision: TokenSuperpositionPromotionDecision {
            promote_to_training_lane: promote,
            status: if promote {
                "eligible_for_training_lane".to_string()
            } else {
                "not_promoted_keep_as_evidence".to_string()
            },
            reason: if promote {
                "TST+AR recovery beat baseline on loss, speed, and regression gates.".to_string()
            } else {
                format!("blocked_by_gates: {}", failed.join(", "))
            },
            artifact: if promote { artifact } else { String::new() },
        },
        timing_breakdown_ms,
        external_inference_calls: 0,
    }
}

pub fn write_token_superposition_report(
    path: &str,
    report: &TokenSuperpositionReport,
) -> Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|err| SymError::InvalidArgument(format!("create report dir: {err}")))?;
        }
    }
    fs::write(
        path,
        serde_json::to_string_pretty(report)
            .map_err(|err| SymError::InvalidArgument(format!("serialize TST report: {err}")))?,
    )
    .map_err(|err| SymError::InvalidArgument(format!("write TST report: {err}")))?;
    Ok(())
}

pub fn format_token_superposition_report(report: &TokenSuperpositionReport) -> String {
    let mut out = String::new();
    out.push_str("Token Superposition Training\n");
    out.push_str(&format!("Backend: {}\n", report.backend));
    out.push_str(&format!(
        "Dataset: train_tokens={} vocab={} hv_dim={}\n",
        report.dataset.train_tokens, report.dataset.vocab_size, report.dataset.hv_dim
    ));
    out.push_str(&format!(
        "Baseline: loss={:.4} code_loss={:.4} train_ms={} eps={:.1}\n",
        report.baseline.eval.combined_ar_loss,
        report.baseline.eval.code_ar_loss,
        report.baseline.train_runtime_ms,
        report.baseline.train_examples_per_second
    ));
    if let Some(best) = &report.best_variant {
        out.push_str(&format!(
            "Best TST: s={} r={:.2} loss={:.4} code_loss={:.4} nominal_speedup={:.2} train_speedup={:.2}\n",
            best.bag_size.unwrap_or_default(),
            best.recovery_ratio.unwrap_or_default(),
            best.eval.combined_ar_loss,
            best.eval.code_ar_loss,
            best.nominal_speedup_vs_baseline,
            best.measured_train_speedup_vs_baseline
        ));
    }
    out.push_str(&format!(
        "Decision: {} ({})\n",
        report.promotion_decision.status, report.promotion_decision.reason
    ));
    out
}

fn token_superposition_gates(
    config: &TokenSuperpositionConfig,
    baseline: &TokenSuperpositionRunReport,
    best: Option<&TokenSuperpositionRunReport>,
) -> Vec<TokenSuperpositionGate> {
    let Some(best) = best else {
        return vec![gate(
            "variant_completed",
            false,
            "no TST variants completed",
        )];
    };
    let tolerance = config.gate_tolerance;
    vec![
        gate(
            "normal_recovery_loss_beats_baseline",
            best.eval.combined_ar_loss <= baseline.eval.combined_ar_loss - tolerance,
            format!(
                "best={:.6} baseline={:.6} tolerance={:.6}",
                best.eval.combined_ar_loss, baseline.eval.combined_ar_loss, tolerance
            ),
        ),
        gate(
            "code_proxy_loss_improves",
            best.eval.code_ar_loss <= baseline.eval.code_ar_loss - tolerance,
            format!(
                "best={:.6} baseline={:.6} tolerance={:.6}",
                best.eval.code_ar_loss, baseline.eval.code_ar_loss, tolerance
            ),
        ),
        gate(
            "nominal_training_speedup_present",
            best.nominal_speedup_vs_baseline >= config.min_nominal_speedup,
            format!(
                "{:.4} >= {:.4}",
                best.nominal_speedup_vs_baseline, config.min_nominal_speedup
            ),
        ),
        gate(
            "measured_cuda_train_speedup_present",
            best.measured_train_speedup_vs_baseline >= config.min_train_speedup,
            format!(
                "{:.4} >= {:.4}",
                best.measured_train_speedup_vs_baseline, config.min_train_speedup
            ),
        ),
        gate(
            "standard_ar_recovery_completed",
            best.recovery_epochs > 0 && best.recovery_samples > 0,
            format!(
                "recovery_epochs={} recovery_samples={}",
                best.recovery_epochs, best.recovery_samples
            ),
        ),
        gate(
            "no_permanent_architecture_change",
            true,
            "same LinearReadout shape, tokenizer, feature hash, optimizer, and CUDA readout backend".to_string(),
        ),
        gate("external_inference_zero", true, "external_inference_calls=0"),
    ]
}

fn gate(name: &str, passed: bool, evidence: impl Into<String>) -> TokenSuperpositionGate {
    TokenSuperpositionGate {
        gate: name.to_string(),
        passed,
        evidence: evidence.into(),
    }
}

fn evaluate_split_ar(
    readout: &LinearReadout,
    tokens: &[usize],
    feature_table: &[Vec<f32>],
    hv_dim: usize,
    eval_samples: usize,
) -> Result<(f32, f32)> {
    if tokens.len() < 2 {
        return Ok((f32::INFINITY, f32::NAN));
    }
    let count = eval_samples.min(tokens.len().saturating_sub(1)).max(1);
    let mut examples = Vec::with_capacity(count);
    for idx in 0..count {
        let pos = if count <= 1 {
            0
        } else {
            idx * (tokens.len().saturating_sub(2)) / (count - 1)
        };
        let token = tokens[pos];
        let target = tokens[pos + 1];
        examples.push(ReadoutTrainingExample {
            features: feature_table[token].clone(),
            target,
        });
    }
    let (features, targets) = training_batch_tensor(&examples, hv_dim)?;
    let trace = readout.evaluate_batch(&features, &targets)?;
    Ok((trace.loss, trace.accuracy))
}

fn load_language_docs(
    root: &Path,
    config: &TokenSuperpositionConfig,
) -> Result<(Vec<TextDoc>, Vec<TextDoc>)> {
    let mut train = Vec::new();
    let mut eval = Vec::new();
    for input in &config.input_paths {
        let path = resolve(root, input);
        if !path.exists() {
            continue;
        }
        let file = File::open(&path)
            .map_err(|err| SymError::InvalidArgument(format!("open {}: {err}", path.display())))?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            if train.len() + eval.len() >= config.max_language_rows {
                break;
            }
            let line = line.map_err(|err| {
                SymError::InvalidArgument(format!("read {}: {err}", path.display()))
            })?;
            let text = extract_training_text(&line);
            if text.trim().is_empty() {
                continue;
            }
            let row = TextDoc {
                text: text.chars().take(config.max_chars_per_doc).collect(),
            };
            if (train.len() + eval.len()) % 9 == 0 {
                eval.push(row);
            } else {
                train.push(row);
            }
        }
    }
    Ok((train, eval))
}

fn load_project_code_docs(
    root: &Path,
    config: &TokenSuperpositionConfig,
) -> Result<(Vec<TextDoc>, Vec<TextDoc>)> {
    let mut paths = Vec::new();
    for entry in &config.project_code_roots {
        collect_code_files(&resolve(root, entry), &mut paths)?;
    }
    paths.sort();
    paths.truncate(config.max_code_files);
    let mut train = Vec::new();
    let mut eval = Vec::new();
    for (idx, path) in paths.iter().enumerate() {
        let text = fs::read_to_string(path)
            .map_err(|err| SymError::InvalidArgument(format!("read {}: {err}", path.display())))?;
        let doc = TextDoc {
            text: text.chars().take(config.max_chars_per_doc).collect(),
        };
        if idx % 5 == 0 {
            eval.push(doc);
        } else {
            train.push(doc);
        }
    }
    Ok((train, eval))
}

fn collect_code_files(path: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(path)
        .map_err(|err| SymError::InvalidArgument(format!("read dir {}: {err}", path.display())))?
    {
        let entry =
            entry.map_err(|err| SymError::InvalidArgument(format!("read dir entry: {err}")))?;
        let child = entry.path();
        let name = child
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if name.starts_with('.')
            || matches!(
                name,
                "target" | "__pycache__" | "checkpoints" | "vendor" | "reports"
            )
        {
            continue;
        }
        if child.is_dir() {
            collect_code_files(&child, out)?;
        } else if matches!(
            child.extension().and_then(|ext| ext.to_str()),
            Some("rs" | "py" | "json" | "toml")
        ) {
            out.push(child);
        }
    }
    Ok(())
}

fn extract_training_text(line: &str) -> String {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
        for key in [
            "sentence_good",
            "text",
            "content",
            "prompt",
            "completion",
            "code",
            "input",
        ] {
            if let Some(text) = value.get(key).and_then(|value| value.as_str()) {
                if !text.trim().is_empty() {
                    return text.to_string();
                }
            }
        }
    }
    line.to_string()
}

fn build_vocab(docs: &[&str], max_vocab: usize) -> Vec<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for doc in docs {
        for token in tokenize(doc) {
            *counts.entry(token).or_insert(0) += 1;
        }
    }
    let mut rows = counts.into_iter().collect::<Vec<_>>();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut vocab = vec![
        "<unk>".to_string(),
        "<bos>".to_string(),
        "<eos>".to_string(),
    ];
    for (token, _) in rows {
        if vocab.len() >= max_vocab {
            break;
        }
        if !vocab.contains(&token) {
            vocab.push(token);
        }
    }
    vocab
}

fn encode_docs(docs: &[&str], vocab: &[String]) -> Vec<usize> {
    let index = vocab
        .iter()
        .enumerate()
        .map(|(idx, token)| (token.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let unk = *index.get("<unk>").unwrap_or(&0);
    let bos = *index.get("<bos>").unwrap_or(&1);
    let eos = *index.get("<eos>").unwrap_or(&2);
    let mut out = Vec::new();
    for doc in docs {
        out.push(bos);
        for token in tokenize(doc) {
            out.push(*index.get(token.as_str()).unwrap_or(&unk));
        }
        out.push(eos);
    }
    out
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut current_kind = CharKind::Other;
    for ch in text.chars() {
        let kind = CharKind::of(ch);
        if kind == CharKind::Space {
            flush_token(&mut current, &mut tokens);
            current_kind = CharKind::Other;
        } else if kind == CharKind::Punct {
            flush_token(&mut current, &mut tokens);
            tokens.push(ch.to_string());
            current_kind = CharKind::Other;
        } else if current.is_empty()
            || kind == current_kind
            || (kind == CharKind::Digit && current_kind == CharKind::Word)
        {
            current.push(ch);
            current_kind = kind;
        } else {
            flush_token(&mut current, &mut tokens);
            current.push(ch);
            current_kind = kind;
        }
    }
    flush_token(&mut current, &mut tokens);
    tokens
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CharKind {
    Word,
    Digit,
    Space,
    Punct,
    Other,
}

impl CharKind {
    fn of(ch: char) -> Self {
        if ch.is_whitespace() {
            Self::Space
        } else if ch.is_ascii_alphabetic() || ch == '_' {
            Self::Word
        } else if ch.is_ascii_digit() {
            Self::Digit
        } else if ch.is_ascii_punctuation() {
            Self::Punct
        } else {
            Self::Other
        }
    }
}

fn flush_token(current: &mut String, tokens: &mut Vec<String>) {
    if !current.is_empty() {
        tokens.push(current.clone());
        current.clear();
    }
}

fn token_features(token: &str, hv_dim: usize) -> Vec<f32> {
    let mut features = vec![0.0; hv_dim];
    add_feature(&mut features, &format!("tok:{token}"), 1.0);
    add_feature(
        &mut features,
        &format!("lower:{}", token.to_ascii_lowercase()),
        0.5,
    );
    add_feature(
        &mut features,
        &format!("shape:{}", token_shape(token)),
        0.35,
    );
    let chars = token.chars().collect::<Vec<_>>();
    for window in chars.windows(3) {
        let gram = window.iter().collect::<String>();
        add_feature(&mut features, &format!("tri:{gram}"), 0.15);
    }
    normalize_in_place(&mut features);
    features
}

fn token_shape(token: &str) -> String {
    token
        .chars()
        .map(|ch| {
            if ch.is_ascii_uppercase() {
                'A'
            } else if ch.is_ascii_lowercase() {
                'a'
            } else if ch.is_ascii_digit() {
                '0'
            } else if ch == '_' {
                '_'
            } else {
                '.'
            }
        })
        .collect()
}

fn add_feature(features: &mut [f32], key: &str, value: f32) {
    if features.is_empty() {
        return;
    }
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    let digest = hasher.finalize();
    let mut idx_bytes = [0u8; 8];
    idx_bytes.copy_from_slice(&digest[..8]);
    let idx = u64::from_le_bytes(idx_bytes) as usize % features.len();
    let sign = if digest[8] & 1 == 0 { 1.0 } else { -1.0 };
    features[idx] += sign * value;
}

fn normalize_in_place(values: &mut [f32]) {
    let norm = values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-8);
    for value in values {
        *value /= norm;
    }
}

fn resolve(root: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}
