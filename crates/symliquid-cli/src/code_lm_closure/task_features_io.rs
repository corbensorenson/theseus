// Task loading, visible signatures, feature construction, tokenization, and artifact IO for Code LM closure.
// This keeps data/feature plumbing separate from state-sequence token policy and decoder scoring.

use super::*;

mod artifact_io;
mod visible_signature;
pub(super) use artifact_io::*;
pub(super) use visible_signature::*;

pub(super) fn build_training_examples(tasks: &[CodeTask], vocab: &Vocab) -> Vec<TrainExample> {
    let mut examples = Vec::new();
    for task in tasks {
        let mut prev2 = "<BOS>".to_string();
        let mut prev1 = "<BOS>".to_string();
        let prompt = training_prompt_tokens(task).into_iter().collect::<Vec<_>>();
        let mut tokens = solution_body_tokens(task);
        tokens.push("<EOS>".to_string());
        for (position, token) in tokens.into_iter().enumerate() {
            let target = vocab_id(vocab, &token);
            examples.push(TrainExample {
                prompt_tokens: prompt.clone(),
                category: task.category.clone(),
                prev2: prev2.clone(),
                prev1: prev1.clone(),
                position,
                target,
            });
            prev2 = prev1;
            prev1 = token;
        }
    }
    examples
}

pub(super) fn evaluate_next_token(
    readout: &LinearReadout,
    examples: &[TrainExample],
    hv_dim: usize,
) -> Result<symliquid_core::train::TrainingTrace, Box<dyn std::error::Error>> {
    if examples.is_empty() {
        return Ok(symliquid_core::train::TrainingTrace {
            loss: 0.0,
            accuracy: 0.0,
            grad_norm: 0.0,
        });
    }
    let rows = feature_rows_for_examples_cached(examples, hv_dim);
    let mut targets = Vec::with_capacity(examples.len());
    for example in examples {
        targets.push(example.target);
    }
    let features = Tensor::new(examples.len(), hv_dim, rows)?;
    Ok(readout.evaluate_batch(&features, &targets)?)
}

pub(super) fn feature_rows_for_examples_cached(
    examples: &[TrainExample],
    hv_dim: usize,
) -> Vec<f32> {
    let mut rows = Vec::with_capacity(examples.len() * hv_dim);
    let mut context_cache = std::collections::HashMap::new();
    for example in examples {
        rows.extend(featurize_with_context_cache(
            example,
            hv_dim,
            &mut context_cache,
        ));
    }
    rows
}

pub(super) fn featurize_with_context_cache(
    example: &TrainExample,
    hv_dim: usize,
    context_cache: &mut std::collections::HashMap<String, TokenFeatureStaticContext>,
) -> Vec<f32> {
    let key = token_feature_context_key(&example.prompt_tokens, &example.category);
    let context = context_cache.entry(key).or_insert_with(|| {
        token_feature_static_context(&example.prompt_tokens, &example.category, hv_dim)
    });
    featurize_with_static_context(context, &example.prev2, &example.prev1, example.position)
}

fn token_feature_context_key(prompt_tokens: &[String], category: &str) -> String {
    let mut key = String::with_capacity(category.len() + prompt_tokens.len() * 8);
    key.push_str(category);
    key.push('\u{1f}');
    for token in prompt_tokens {
        key.push_str(token);
        key.push('\u{1e}');
    }
    key
}

#[derive(Debug, Clone)]
pub(super) struct TokenFeatureStaticContext {
    base: Vec<f32>,
    category: String,
    position_cap: usize,
}

pub(super) fn token_feature_static_context(
    prompt_tokens: &[String],
    category: &str,
    hv_dim: usize,
) -> TokenFeatureStaticContext {
    let mut out = vec![0.0; hv_dim];
    let position_cap = if execution_shaped_category(category) {
        192
    } else {
        48
    };
    add_feature(&mut out, &format!("category:{category}"), 1.2);
    for token in prompt_tokens.iter().take(32) {
        add_feature(&mut out, &format!("prompt:{token}"), 0.35);
        add_feature(
            &mut out,
            &format!("category:{category}|prompt:{token}"),
            0.22,
        );
    }
    TokenFeatureStaticContext {
        base: out,
        category: category.to_string(),
        position_cap,
    }
}

pub(super) fn featurize_with_static_context(
    context: &TokenFeatureStaticContext,
    prev2: &str,
    prev1: &str,
    position: usize,
) -> Vec<f32> {
    let mut out = context.base.clone();
    let capped_position = position.min(context.position_cap);
    add_feature(&mut out, &format!("prev1:{prev1}"), 1.5);
    add_feature(&mut out, &format!("prev2:{prev2}"), 0.8);
    add_feature(&mut out, &format!("prev2prev1:{prev2}|{prev1}"), 1.0);
    add_feature(&mut out, &format!("pos:{capped_position}"), 0.65);
    add_feature(
        &mut out,
        &format!("category:{}|pos:{capped_position}", context.category),
        1.25,
    );
    add_feature(
        &mut out,
        &format!("category:{}|prev1:{prev1}", context.category),
        0.95,
    );
    add_feature(
        &mut out,
        &format!("category:{}|prev2prev1:{prev2}|{prev1}", context.category),
        1.1,
    );
    add_feature(
        &mut out,
        &format!("pos:{capped_position}|prev1:{prev1}"),
        0.45,
    );
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

pub(super) fn add_feature(out: &mut [f32], key: &str, weight: f32) {
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

pub(super) fn build_vocab(tasks: &[CodeTask], max_vocab: usize) -> Vocab {
    let mut counts = HashMap::new();
    for task in tasks {
        for token in solution_body_tokens(task) {
            *counts.entry(token).or_insert(0usize) += 1;
        }
    }
    let mut rows = counts.into_iter().collect::<Vec<_>>();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut id_to_token = vec!["<EOS>".to_string(), "<UNK>".to_string()];
    for (token, _) in rows {
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

pub(super) fn extend_vocab_with_visible_contract_tokens(
    vocab: &mut Vocab,
    task_groups: &[&[CodeTask]],
    max_vocab: usize,
) {
    let mut candidates = BTreeSet::new();
    for group in task_groups {
        for task in *group {
            let entry = sanitize_ident(&task.entry_point);
            let signature = visible_signature(&entry, &task.prompt);
            for arg in signature.arg_names {
                if is_identifier(&arg) {
                    candidates.insert(arg);
                }
            }
            for token in decoder_required_constructs(task) {
                if is_identifier(&token) {
                    candidates.insert(token);
                }
            }
            for token in semantic_decoder_v2_plan_hints(task, None) {
                if is_identifier(&token) {
                    candidates.insert(token);
                }
            }
        }
    }
    for token in candidates {
        if vocab.id_to_token.len() >= max_vocab {
            break;
        }
        if vocab.token_to_id.contains_key(&token) {
            continue;
        }
        let id = vocab.id_to_token.len();
        vocab.id_to_token.push(token.clone());
        vocab.token_to_id.insert(token, id);
    }
}

pub(super) fn vocab_id(vocab: &Vocab, token: &str) -> usize {
    *vocab.token_to_id.get(token).unwrap_or(&vocab.unk_id)
}

pub(super) fn build_expression_bank(tasks: &[CodeTask]) -> Vec<ExpressionBankItem> {
    let mut counts: BTreeMap<(String, String), (usize, BTreeSet<String>)> = BTreeMap::new();
    for task in tasks {
        let candidate_expression_eligible = task
            .raw
            .get("candidate_expression_eligible")
            .and_then(|value| value.as_bool())
            .unwrap_or(true);
        if candidate_expression_eligible && useful_expression(&task.solution_expr) {
            let entry = counts
                .entry((task.category.clone(), task.solution_expr.clone()))
                .or_insert_with(|| (0, BTreeSet::new()));
            entry.0 += 1;
            entry.1.extend(prompt_tokens(task));
        }
    }
    let mut rows = counts
        .into_iter()
        .map(|((category, expr), (count, hints))| ExpressionBankItem {
            tokens: tokenize_code(&expr),
            hints,
            category,
            expr,
            count,
        })
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.expr.cmp(&b.expr)));
    rows
}

pub(super) fn load_tasks(path: &Path) -> Result<Vec<CodeTask>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut tasks = Vec::new();
    for raw in read_jsonl(path)? {
        let task_id = string_field(&raw, "task_id");
        let entry_point = string_field(&raw, "entry_point");
        let prompt =
            prompt_with_inferred_signature(&entry_point, &string_field(&raw, "prompt"), &raw);
        if task_id.is_empty() || prompt.is_empty() || entry_point.is_empty() {
            continue;
        }
        let category = inferred_task_category(&raw, &prompt, &entry_point);
        let tags = raw
            .get("tags")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        tasks.push(CodeTask {
            task_id,
            source_task_id: string_field(&raw, "source_task_id"),
            card_id: string_field(&raw, "card_id"),
            source_id: string_field(&raw, "source_id"),
            split: string_field(&raw, "split"),
            category,
            prompt,
            entry_point,
            solution_expr: string_field(&raw, "solution_expr"),
            solution_body: string_field(&raw, "solution_body"),
            tags,
            benchmark_evidence_level: string_field(&raw, "benchmark_evidence_level"),
            raw,
        });
    }
    Ok(tasks)
}

pub(super) fn inferred_task_category(raw: &Value, prompt: &str, entry_point: &str) -> String {
    let declared = string_field(raw, "category").trim().to_string();
    if useful_declared_category(&declared) {
        return declared;
    }
    if let Some(contract_category) = raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("category"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| useful_declared_category(value))
    {
        return contract_category.to_string();
    }

    let source_task_id = string_field(raw, "source_task_id");
    let text = format!("{declared}\n{source_task_id}\n{entry_point}\n{prompt}").to_lowercase();
    let compact = text
        .chars()
        .filter(|ch| !matches!(ch, '_' | '-' | ' '))
        .collect::<String>();
    let entry = entry_point.to_lowercase();

    if body_has_any(
        &text,
        &[
            "pairplot",
            "dataframe",
            "pandas",
            "minmaxscaler",
            "box plot",
            "plot",
        ],
    ) {
        return "dataframe_transform".to_string();
    }
    if body_has_any(&text, &["csv", "comma separated", "comma-separated"]) {
        return "csv_command_outputs".to_string();
    }
    if body_has_any(&text, &["json"]) {
        return "json_extract_field".to_string();
    }
    if body_has_any(&text, &["urlencode", "url encode", "query string"]) {
        return "urlencode_payload".to_string();
    }
    if body_has_any(&text, &["zip", "zipfile"]) {
        return "zip_flat_directory".to_string();
    }
    if body_has_any(&text, &["tar", "tarfile", "log backup", "logs"]) {
        return "log_backup_tar".to_string();
    }
    if body_has_any(
        &text,
        &["directory", "folder", "path", "file path", "os.listdir"],
    ) {
        return "file_path_transform".to_string();
    }
    if body_has_any(
        &text,
        &["platform", "system info", "architecture", "memory usage"],
    ) {
        return "system_info_dict".to_string();
    }
    if entry.contains("add") && body_has_any(&text, &["two numbers", "add numbers", "sum"]) {
        return "add_numbers".to_string();
    }
    if entry.contains("same") && body_has_any(&text, &["same chars", "same characters"]) {
        return "same_chars".to_string();
    }
    if body_has_any(&text, &["bracket", "parentheses", "parenthesis"]) {
        return "balanced_brackets_simple".to_string();
    }
    if entry.contains("sort_sublists") || body_has_any(&text, &["sort sublists", "sort nested"]) {
        return "sort_list".to_string();
    }
    if compact.contains("matrixsum") || body_has_any(&text, &["matrix sum", "sum matrix"]) {
        return "nested_flat_sum".to_string();
    }
    if compact.contains("minimumrightshifts") || body_has_any(&text, &["right shifts", "rotate"]) {
        return "algorithmic_planning".to_string();
    }
    if body_has_any(
        &text,
        &[
            "subarray",
            "subsequence",
            "triplet",
            "xor",
            "permutation",
            "minimum",
            "maximum",
            "count ways",
            "operations",
            "groups",
            "strong pair",
        ],
    ) {
        return "algorithmic_planning".to_string();
    }
    if body_has_any(&text, &["palindrome"]) {
        return "palindrome".to_string();
    }
    "general_semantics".to_string()
}

pub(super) fn useful_declared_category(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty()
        && !matches!(
            trimmed,
            "general" | "general_semantics" | "unknown" | "none" | "misc"
        )
}

pub(super) fn prompt_with_inferred_signature(
    entry_point: &str,
    prompt: &str,
    raw: &Value,
) -> String {
    let entry = sanitize_ident(entry_point);
    if entry.is_empty() || parse_visible_signature(&entry, prompt).is_some() {
        return prompt.to_string();
    }
    let arg_count = inferred_visible_arg_count(raw);
    let arg_names = ["data", "other", "third", "fourth", "fifth", "sixth"];
    let args = arg_names
        .iter()
        .take(arg_count.min(arg_names.len()))
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    format!("def {entry}({args}):\n{prompt}")
}

pub(super) fn inferred_visible_arg_count(raw: &Value) -> usize {
    if let Some(count) = raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("visible_arg_count_hint"))
        .and_then(Value::as_u64)
    {
        return count.min(6) as usize;
    }
    let body = string_field(raw, "solution_body").to_lowercase();
    if body_mentions_token(&body, "third") || body_mentions_token(&body, "extra") {
        return 3;
    }
    if body_mentions_token(&body, "other") {
        return 2;
    }
    if body_mentions_token(&body, "data") {
        return 1;
    }
    inferred_arg_count_from_tests(raw).unwrap_or(0).min(6)
}

pub(super) fn inferred_arg_count_from_tests(raw: &Value) -> Option<usize> {
    let entry = sanitize_ident(&string_field(raw, "entry_point"));
    if entry.is_empty() {
        return None;
    }
    let tests = string_field(raw, "tests");
    let needle = format!("{entry}(");
    let start = tests.find(&needle)? + needle.len();
    let chars = tests[start..].chars().collect::<Vec<_>>();
    let mut depth = 0i32;
    let mut count = 0usize;
    let mut saw_token = false;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for ch in chars {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == q {
                quote = None;
            }
            saw_token = true;
            continue;
        }
        match ch {
            '\'' | '"' => {
                quote = Some(ch);
                saw_token = true;
            }
            '(' | '[' | '{' => {
                depth += 1;
                saw_token = true;
            }
            ')' if depth == 0 => {
                return Some(if saw_token { count + 1 } else { 0 });
            }
            ')' | ']' | '}' => {
                depth -= 1;
                saw_token = true;
            }
            ',' if depth == 0 => {
                count += 1;
            }
            ch if !ch.is_whitespace() => {
                saw_token = true;
            }
            _ => {}
        }
    }
    None
}

pub(super) fn load_sts_streams(path: &Path) -> Result<StsStreamMap, Box<dyn std::error::Error>> {
    if path.as_os_str().is_empty() || !path.exists() {
        return Ok(HashMap::new());
    }
    let mut out: StsStreamMap = HashMap::new();
    for raw in read_jsonl(path)? {
        let task_id = string_field(&raw, "task_id");
        if task_id.is_empty() {
            continue;
        }
        let streams = raw
            .get("streams")
            .and_then(Value::as_object)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|(key, value)| {
                        let text = value.as_str().unwrap_or_default().trim().to_string();
                        if text.is_empty() {
                            None
                        } else {
                            Some((key.clone(), text))
                        }
                    })
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        let public_solutions = raw
            .get("public_benchmark_solutions_included")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !streams.is_empty() && !public_solutions {
            out.insert(task_id, streams);
        }
    }
    Ok(out)
}

pub(super) fn merge_sts_decoder_control_policy_for_tasks(
    sts_streams: &mut StsStreamMap,
    private_tasks: &[CodeTask],
    public_tasks: &[CodeTask],
    path: &Path,
) -> Result<usize, Box<dyn std::error::Error>> {
    if std::env::var("THESEUS_CODE_LM_DISABLE_STS_DECODER_CONTROL_POLICY")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            value == "1" || value == "true" || value == "on"
        })
        .unwrap_or(false)
    {
        return Ok(0);
    }
    if path.as_os_str().is_empty() || !path.exists() {
        return Ok(0);
    }
    let mut fragments = Vec::new();
    for raw in read_jsonl(path)? {
        let public_solutions = raw
            .get("public_benchmark_solutions_included")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || raw
                .get("canonical_solution_exported")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            || raw
                .get("public_tests_included")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            || raw
                .get("raw_public_prompt_or_tests_copied")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        if public_solutions {
            continue;
        }
        let objective = string_field(&raw, "objective");
        let answer = string_field(&raw, "answer");
        if objective.is_empty() && answer.is_empty() {
            continue;
        }
        let families = raw
            .get("targeted_capability_families")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        fragments.push(format!(
            "sts_decoder_control_policy objective={}; target_families={}; {}",
            objective,
            families.join(", "),
            answer
        ));
    }
    if fragments.is_empty() {
        return Ok(0);
    }
    let merged_control = fragments.join("\n");
    let mut applied = 0usize;
    for task in private_tasks.iter().chain(public_tasks.iter()) {
        let Some(task_streams) = sts_streams.get_mut(&task.task_id) else {
            continue;
        };
        task_streams.insert("decoder_control_stream".to_string(), merged_control.clone());
        if let Some(private_v3_stream) = private_residual_v3_decoder_contract_stream(task) {
            task_streams.insert(
                "private_residual_v3_decoder_contract_stream".to_string(),
                private_v3_stream,
            );
        }
        applied = applied.saturating_add(1);
    }
    Ok(applied)
}

fn private_residual_v3_decoder_contract_stream(task: &CodeTask) -> Option<String> {
    let contract = task
        .raw
        .get("decoder_contract")
        .and_then(Value::as_object)?;
    let policy = contract.get("policy").and_then(Value::as_str).unwrap_or("");
    if task.card_id != "private_residual_repair_v3"
        || !task
            .benchmark_evidence_level
            .contains("private_residual_repair_v3_generated_only")
        || policy != "project_theseus_decoder_contract_v3_private_residual_repair"
    {
        return None;
    }
    let residual_family = task
        .raw
        .get("targeted_private_residual_family_v3")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let required = decoder_required_constructs(task)
        .into_iter()
        .collect::<Vec<_>>()
        .join(", ");
    let hints = decoder_contract_generation_hints(task)
        .into_iter()
        .collect::<Vec<_>>()
        .join(", ");
    let skeleton_bias = contract
        .get("skeleton_bias")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    Some(format!(
        "private_residual_v3_decoder_contract_stream \
         category={}; residual_family={}; return_shape={}; type_family={}; \
         required_constructs={}; generation_hints={}; skeleton_bias={}; \
         route=sts_conditioned_semantic_adapter; \
         policy=private_generated_curriculum_only_no_public_tests_or_solutions; \
         decoder_focus=loop branch local state return_shape interface contract parse split stdin \
         string index selection minimum maximum frequency count threshold graph interval nested flatten",
        task.category,
        residual_family,
        decoder_return_shape(task),
        decoder_type_family(task),
        required,
        hints,
        skeleton_bias,
    ))
}

pub(super) fn prompt_tokens(task: &CodeTask) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for token in tokenize_code(&format!(
        "{} {} {} {}",
        task.entry_point,
        task.category,
        task.prompt,
        task.tags.join(" ")
    )) {
        if is_identifier(&token) {
            add_identifier_variants(&mut out, &token);
        }
    }
    add_contract_prompt_tokens(&mut out, task);
    out
}

pub(super) fn add_contract_prompt_tokens(out: &mut BTreeSet<String>, task: &CodeTask) {
    let shape = decoder_return_shape(task);
    if shape != "unknown" {
        out.insert(format!("return_shape:{shape}"));
        out.insert(format!("category_return_shape:{}:{shape}", task.category));
    }
    let family = decoder_type_family(task);
    if family != "unknown" {
        out.insert(format!("type_family:{family}"));
        out.insert(format!("category_type_family:{}:{family}", task.category));
    }
    if let Some(arg_count) = category_expected_arg_count(task) {
        out.insert(format!("arg_count:{arg_count}"));
        out.insert(format!("category_arg_count:{}:{arg_count}", task.category));
    }
    for (idx, arg_name) in visible_signature_arg_names(task)
        .into_iter()
        .enumerate()
        .take(6)
    {
        add_identifier_variants(out, &arg_name);
        out.insert(format!("arg_name:{idx}:{arg_name}"));
        out.insert(format!(
            "category_arg_name:{}:{idx}:{arg_name}",
            task.category
        ));
    }
    for (idx, kind) in visible_arg_kinds(&task.entry_point, &task.prompt)
        .into_iter()
        .enumerate()
        .take(6)
    {
        let kind = format!("{:?}", kind).to_lowercase();
        out.insert(format!("arg_kind:{idx}:{kind}"));
        out.insert(format!("category_arg_kind:{}:{idx}:{kind}", task.category));
    }
    for required in decoder_required_constructs(task).into_iter().take(12) {
        out.insert(format!("requires:{required}"));
        out.insert(format!("category_requires:{}:{required}", task.category));
    }
    for hint in semantic_decoder_v2_plan_hints(task, None)
        .into_iter()
        .take(32)
    {
        out.insert(format!("plan:{hint}"));
        out.insert(format!("category_plan:{}:{hint}", task.category));
    }
}

pub(super) fn training_prompt_tokens(task: &CodeTask) -> BTreeSet<String> {
    let mut tokens = prompt_tokens(task);
    add_stream_evidence_tokens(&mut tokens, "solver_stream", &task.solution_expr);
    add_stream_evidence_tokens(&mut tokens, "body_stream", &task.solution_body_text());
    add_stream_evidence_tokens(&mut tokens, "patch_stream", &task.solution_body_text());
    add_stream_evidence_tokens(&mut tokens, "critic_stream", &task.solution_body_text());
    tokens
}

pub(super) fn prompt_tokens_with_sts(
    task: &CodeTask,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> BTreeSet<String> {
    let mut tokens = prompt_tokens(task);
    if let Some(streams) = sts_streams {
        for (stream, text) in streams {
            add_stream_evidence_tokens(&mut tokens, stream, text);
            for token in tokenize_code(&format!("{stream} {text}")) {
                if is_identifier(&token) {
                    add_identifier_variants(&mut tokens, &token);
                }
            }
        }
    }
    tokens
}

pub(super) fn add_stream_evidence_tokens(out: &mut BTreeSet<String>, stream: &str, text: &str) {
    for token in tokenize_code(text) {
        let lowered = token.to_lowercase();
        if is_identifier(&token) {
            add_identifier_variants(out, &token);
            for part in token.split('_') {
                let part = part.to_lowercase();
                if part.len() >= 2 {
                    out.insert(format!("{stream}:{part}"));
                }
            }
        }
        if !lowered.trim().is_empty() {
            out.insert(format!("{stream}:{lowered}"));
        }
    }
}

pub(super) fn add_identifier_variants(out: &mut BTreeSet<String>, token: &str) {
    let lowered = token.to_lowercase();
    if lowered.is_empty() {
        return;
    }
    insert_token_variant(out, &lowered);
    for part in lowered.split('_') {
        insert_token_variant(out, part);
    }
    for part in camel_parts(token) {
        insert_token_variant(out, &part.to_lowercase());
    }
}

pub(super) fn insert_token_variant(out: &mut BTreeSet<String>, token: &str) {
    let cleaned = token.trim_matches('_');
    if cleaned.len() < 2 {
        return;
    }
    out.insert(cleaned.to_string());
    if cleaned.len() > 3 && cleaned.ends_with('s') {
        out.insert(cleaned.trim_end_matches('s').to_string());
    }
}

pub(super) fn camel_parts(token: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut previous_lower = false;
    for ch in token.chars() {
        if ch == '_' {
            if !current.is_empty() {
                parts.push(current.clone());
                current.clear();
            }
            previous_lower = false;
            continue;
        }
        if ch.is_ascii_uppercase() && previous_lower && !current.is_empty() {
            parts.push(current.clone());
            current.clear();
        }
        previous_lower = ch.is_ascii_lowercase();
        current.push(ch);
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

pub(super) fn render_candidate_body(task: &CodeTask, body: &str) -> String {
    let entry = sanitize_ident(&task.entry_point);
    let signature = if broad_private_generalization_varargs_signature_required(task) {
        VisibleSignature {
            header: format!("def {entry}(*args):"),
            arg_names: Vec::new(),
            uses_varargs: true,
        }
    } else {
        visible_signature(&entry, &task.prompt)
    };
    let import_prelude = visible_prompt_import_prelude(&task.prompt);
    let body = canonicalize_candidate_body_aliases(body, &signature);
    let mut body_lines = Vec::new();
    if signature.uses_varargs || body_needs_signature_aliases(&body) {
        for line in signature.alias_lines() {
            body_lines.push(format!("        {line}"));
        }
    }
    for line in body.lines() {
        let trimmed_end = line.trim_end();
        if trimmed_end.trim().is_empty() {
            continue;
        }
        body_lines.push(format!("        {trimmed_end}"));
    }
    if body_lines.is_empty() {
        body_lines.push("        return None".to_string());
    }
    let rendered_body = body_lines.join("\n");
    let mut prelude = vec!["from typing import *".to_string()];
    prelude.extend(import_prelude);
    prelude.dedup();
    format!(
        "{}\n\n{}\n{rendered_body}\n",
        prelude.join("\n"),
        signature.header
    )
}

fn broad_private_generalization_varargs_signature_required(task: &CodeTask) -> bool {
    let policy = task
        .raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("policy"))
        .and_then(Value::as_str)
        .unwrap_or("");
    task.card_id == "broad_private_generalization_ladder_v1"
        && task
            .benchmark_evidence_level
            .contains("broad_private_generalization_ladder_v1_generated_only")
        && policy == "project_theseus_decoder_contract_v1_broad_private_generalization"
}

pub(super) fn canonicalize_candidate_body_aliases(
    body: &str,
    signature: &VisibleSignature,
) -> String {
    if signature.uses_varargs {
        return body.to_string();
    }
    let visible_args = signature.arg_names.iter().cloned().collect::<BTreeSet<_>>();
    let mut replacements = Vec::new();
    if let Some(first) = signature.arg_names.first() {
        push_arg_alias_replacements(
            body,
            &visible_args,
            &mut replacements,
            first,
            &[
                "data",
                "items",
                "nums",
                "numbers",
                "arr",
                "array",
                "values",
                "xs",
                "lst",
                "payload",
                "text",
                "string",
                "s",
                "path",
                "file_path",
                "filename",
                "config",
            ],
        );
    }
    if let Some(second) = signature.arg_names.get(1) {
        push_arg_alias_replacements(
            body,
            &visible_args,
            &mut replacements,
            second,
            &[
                "other",
                "target",
                "needle",
                "pattern",
                "threshold",
                "limit",
                "k",
                "right",
                "second",
                "b",
                "list2",
                "l2",
                "output_dir",
                "dest",
                "destination",
                "replacement",
                "sep",
                "delimiter",
            ],
        );
    }
    if let Some(third) = signature.arg_names.get(2) {
        push_arg_alias_replacements(
            body,
            &visible_args,
            &mut replacements,
            third,
            &[
                "third",
                "default",
                "fallback",
                "replacement",
                "sep",
                "delimiter",
            ],
        );
    }
    let mut out = body.to_string();
    let mut sentinels = Vec::new();
    for (idx, (from, to)) in replacements
        .into_iter()
        .filter(|(from, to)| from != to)
        .enumerate()
    {
        let sentinel = format!("__THESEUS_ARG_ALIAS_{idx}__");
        out = replace_identifier_token(&out, &from, &sentinel);
        sentinels.push((sentinel, to));
    }
    for (sentinel, to) in sentinels {
        out = replace_identifier_token(&out, &sentinel, &to);
    }
    out
}

fn body_needs_signature_aliases(body: &str) -> bool {
    identifier_token_present(body, "args") || identifier_token_present(body, "extra")
}

#[allow(dead_code)]
pub(super) fn featurize(example: &TrainExample, hv_dim: usize) -> Vec<f32> {
    let context = token_feature_static_context(&example.prompt_tokens, &example.category, hv_dim);
    featurize_with_static_context(&context, &example.prev2, &example.prev1, example.position)
}

pub(super) fn push_arg_alias_replacements(
    body: &str,
    visible_args: &BTreeSet<String>,
    replacements: &mut Vec<(String, String)>,
    target: &str,
    aliases: &[&str],
) {
    if identifier_token_present(body, target) {
        return;
    }
    for alias in aliases {
        if *alias == target
            || visible_args.contains(*alias)
            || alias_looks_like_local_binding(body, alias)
        {
            continue;
        }
        if identifier_token_present(body, alias) {
            replacements.push(((*alias).to_string(), target.to_string()));
        }
    }
}

pub(super) fn alias_looks_like_local_binding(body: &str, alias: &str) -> bool {
    let lowered = body.to_lowercase();
    let alias_lower = alias.to_lowercase();
    lowered.contains(&format!("for {alias_lower} in"))
        || lowered.starts_with(&format!("{alias_lower} ="))
        || lowered.starts_with(&format!("{alias_lower} +="))
        || lowered.contains(&format!("\n{alias_lower} ="))
        || lowered.contains(&format!("\n{alias_lower} +="))
        || lowered.contains(&format!("\n{alias_lower}."))
        || lowered.contains(&format!(" as {alias_lower}"))
}

pub(super) fn identifier_token_present(text: &str, token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    let chars = text.chars().collect::<Vec<_>>();
    let token_chars = token.chars().collect::<Vec<_>>();
    (0..chars.len()).any(|idx| identifier_token_at(&chars, idx, &token_chars))
}

pub(super) fn replace_identifier_token(text: &str, token: &str, replacement: &str) -> String {
    if token == replacement || token.is_empty() {
        return text.to_string();
    }
    let chars = text.chars().collect::<Vec<_>>();
    let token_chars = token.chars().collect::<Vec<_>>();
    let mut out = String::new();
    let mut idx = 0usize;
    while idx < chars.len() {
        if identifier_token_at(&chars, idx, &token_chars) {
            out.push_str(replacement);
            idx += token_chars.len();
        } else {
            out.push(chars[idx]);
            idx += 1;
        }
    }
    out
}

pub(super) fn identifier_token_at(chars: &[char], idx: usize, token_chars: &[char]) -> bool {
    if token_chars.is_empty() || idx + token_chars.len() > chars.len() {
        return false;
    }
    for (offset, expected) in token_chars.iter().enumerate() {
        if chars[idx + offset] != *expected {
            return false;
        }
    }
    let before_ok = idx == 0 || !is_identifier_char(chars[idx - 1]);
    let after_idx = idx + token_chars.len();
    let after_ok = after_idx >= chars.len() || !is_identifier_char(chars[after_idx]);
    before_ok && after_ok
}

pub(super) fn body_from_expression(expression: &str) -> String {
    format!("return {}", expression.trim())
}

pub(super) fn solution_body_tokens(task: &CodeTask) -> Vec<String> {
    tokenize_body(&task.solution_body_text())
}

pub(super) fn tokenize_body(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current_indent = 0usize;
    let mut first_line = true;
    for raw_line in text.lines() {
        if raw_line.trim().is_empty() {
            continue;
        }
        if !first_line {
            tokens.push("<NL>".to_string());
        }
        first_line = false;
        let leading_spaces = raw_line.chars().take_while(|ch| *ch == ' ').count();
        let indent = leading_spaces / 4;
        while current_indent < indent {
            tokens.push("<INDENT>".to_string());
            current_indent += 1;
        }
        while current_indent > indent {
            tokens.push("<DEDENT>".to_string());
            current_indent -= 1;
        }
        tokens.extend(tokenize_code(raw_line.trim()));
    }
    while current_indent > 0 {
        tokens.push("<NL>".to_string());
        tokens.push("<DEDENT>".to_string());
        current_indent -= 1;
    }
    tokens
}

pub(super) fn join_body_tokens(tokens: &[String]) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut indent = 0usize;
    for token in tokens {
        match token.as_str() {
            "<EOS>" => break,
            "<UNK>" => continue,
            "<NL>" => {
                if !current.trim().is_empty() {
                    lines.push(format!("{}{}", "    ".repeat(indent), current.trim_end()));
                }
                current.clear();
            }
            "<INDENT>" => {
                if current.trim().is_empty() {
                    indent = (indent + 1).min(4);
                }
            }
            "<DEDENT>" => {
                if current.trim().is_empty() {
                    indent = indent.saturating_sub(1);
                }
            }
            _ => append_code_token(&mut current, token),
        }
    }
    if !current.trim().is_empty() {
        lines.push(format!("{}{}", "    ".repeat(indent), current.trim_end()));
    }
    lines.join("\n")
}

pub(super) fn append_code_token(out: &mut String, token: &str) {
    let no_space_before = [")", "]", "}", ",", ".", ":", "<NL>", "<DEDENT>"].contains(&token);
    let no_space_after_prev =
        out.ends_with('(') || out.ends_with('[') || out.ends_with('{') || out.ends_with('.');
    if !out.is_empty() && !no_space_before && !no_space_after_prev {
        out.push(' ');
    }
    out.push_str(token);
}

pub(super) fn tokenize_code(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let chars = text.chars().collect::<Vec<_>>();
    let mut i = 0usize;
    while i < chars.len() {
        let ch = chars[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch == '"' || ch == '\'' {
            let quote = ch;
            let start = i;
            i += 1;
            let mut escaped = false;
            while i < chars.len() {
                if !escaped && chars[i] == quote {
                    i += 1;
                    break;
                }
                if chars[i] == '\\' {
                    escaped = !escaped;
                    i += 1;
                    continue;
                }
                escaped = false;
                i += 1;
            }
            let literal = chars[start..i.min(chars.len())].iter().collect::<String>();
            if useful_string_literal_token(&literal) {
                tokens.push(literal);
            } else {
                tokens.push("\"STR\"".to_string());
            }
            continue;
        }
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = i;
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect::<String>());
            continue;
        }
        if ch.is_ascii_digit() {
            let start = i;
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            let literal = chars[start..i].iter().collect::<String>();
            tokens.push(normalize_numeric_literal(&literal));
            continue;
        }
        if i + 2 < chars.len() {
            let three = format!("{}{}{}", chars[i], chars[i + 1], chars[i + 2]);
            if ["//=", "**="].contains(&three.as_str()) {
                tokens.push(three);
                i += 3;
                continue;
            }
        }
        if i + 1 < chars.len() {
            let two = format!("{}{}", chars[i], chars[i + 1]);
            if [
                "==", "!=", "<=", ">=", "//", "**", "->", "+=", "-=", "*=", "/=", "%=", "&=", "|=",
            ]
            .contains(&two.as_str())
            {
                tokens.push(two);
                i += 2;
                continue;
            }
        }
        tokens.push(ch.to_string());
        i += 1;
    }
    tokens
}

pub(super) fn normalize_numeric_literal(literal: &str) -> String {
    if literal.contains('.') {
        let normalized = literal.trim_start_matches('+');
        if matches!(
            normalized,
            "0.5" | ".5" | "3.14" | "3.1415" | "3.14159" | "3.141592653589793"
        ) {
            return if normalized == ".5" {
                "0.5".to_string()
            } else {
                normalized.to_string()
            };
        }
        return "0".to_string();
    }
    match literal.parse::<u64>() {
        Ok(value) if value <= 64 => value.to_string(),
        _ => "0".to_string(),
    }
}

pub(super) fn useful_string_literal_token(literal: &str) -> bool {
    if literal.len() > 48 || literal.len() < 2 {
        return false;
    }
    let body = &literal[1..literal.len().saturating_sub(1)];
    if body.contains('\n') || body.contains('\r') || body.contains('\t') {
        return false;
    }
    body.chars().all(|ch| ch.is_ascii_graphic() || ch == ' ')
}

pub(super) fn useful_expression(expr: &str) -> bool {
    let trimmed = expr.trim();
    if trimmed.is_empty() || trimmed.len() > 220 {
        return false;
    }
    if matches!(
        trimmed,
        "[]" | "{}" | "()" | "''" | "\"\"" | "None" | "True" | "False" | "set()"
    ) {
        return true;
    }
    if !trimmed.chars().any(|ch| ch.is_ascii_alphanumeric()) {
        return false;
    }
    let lowered = trimmed.to_lowercase();
    let blocked = [
        "open(",
        "exec(",
        "eval(",
        "__",
        "import ",
        "assert ",
        "sys.",
        "subprocess",
        "os.",
        "task_id",
        "canonical",
        "solution",
        "candidate generated",
        "private curriculum",
        "emit ",
        "full python body",
        "using private concept pressure",
        "student decoder",
    ];
    !blocked.iter().any(|token| lowered.contains(token)) && delimiters_balanced(trimmed)
}

pub(super) fn useful_generated_expression(expr: &str) -> bool {
    if !useful_expression(expr) {
        return false;
    }
    let trimmed = expr.trim();
    if trimmed.eq_ignore_ascii_case("return") {
        return false;
    }
    if trimmed.contains(';') || trimmed.contains('\n') || contains_assignment_operator(trimmed) {
        return false;
    }
    if contains_top_level_for_in_tail(trimmed) {
        return false;
    }
    let lowered = trimmed.to_lowercase();
    if invalid_generated_expression_syntax(trimmed) {
        return false;
    }
    if [
        "if", "else", "for", "in", "and", "or", "not", "lambda", "return",
    ]
    .iter()
    .any(|token| lowered == *token || lowered.ends_with(&format!(" {token}")))
    {
        return false;
    }
    if trimmed.ends_with(|ch: char| "+-*/%<>=!&|.,:".contains(ch)) {
        return false;
    }
    if tokenize_code(trimmed).iter().any(|token| token == "return") {
        return false;
    }
    if lowered.contains("\"str\":") {
        return false;
    }
    true
}

pub(super) fn invalid_generated_expression_syntax(expr: &str) -> bool {
    let tokens = tokenize_code(expr);
    if tokens.is_empty() {
        return true;
    }
    let mut saw_ternary_if_without_comprehension = false;
    let mut saw_else_after_ternary_if = false;
    let mut saw_for_before_if = false;
    for (idx, token) in tokens.iter().enumerate() {
        let prev = idx.checked_sub(1).and_then(|pos| tokens.get(pos));
        let next = tokens.get(idx + 1);
        if token == "for" {
            saw_for_before_if = true;
        }
        if token == "if" && !saw_for_before_if {
            saw_ternary_if_without_comprehension = true;
        }
        if token == "else" && saw_ternary_if_without_comprehension {
            saw_else_after_ternary_if = true;
        }
        if matches!(token.as_str(), "and" | "or") {
            if idx == 0 || next.is_none() {
                return true;
            }
            if next.is_some_and(|item| {
                expression_boundary_token(item) || infix_expression_operator(item)
            }) {
                return true;
            }
            if prev
                .is_some_and(|item| infix_expression_operator(item) || item == "(" || item == ",")
            {
                return true;
            }
        }
        if comparison_expression_operator(token) {
            if idx == 0 || next.is_none() {
                return true;
            }
            if next.is_some_and(|item| {
                expression_boundary_token(item)
                    || infix_expression_operator(item)
                    || matches!(item.as_str(), "and" | "or" | "else")
            }) {
                return true;
            }
            if prev.is_some_and(|item| {
                matches!(
                    item.as_str(),
                    "and" | "or" | "if" | "else" | "(" | "," | ":"
                )
            }) {
                return true;
            }
        }
        if arithmetic_expression_operator(token) {
            if next.is_none() || next.is_some_and(|item| expression_boundary_token(item)) {
                return true;
            }
        }
    }
    saw_ternary_if_without_comprehension && !saw_else_after_ternary_if
}

pub(super) fn expression_boundary_token(token: &str) -> bool {
    matches!(token, ")" | "]" | "}" | "," | ":")
}

pub(super) fn comparison_expression_operator(token: &str) -> bool {
    matches!(token, "<" | ">" | "<=" | ">=" | "==" | "!=" | "in" | "is")
}

pub(super) fn arithmetic_expression_operator(token: &str) -> bool {
    matches!(token, "+" | "*" | "/" | "//" | "%" | "**" | "|" | "&" | "^")
}

pub(super) fn infix_expression_operator(token: &str) -> bool {
    matches!(token, "and" | "or")
        || comparison_expression_operator(token)
        || arithmetic_expression_operator(token)
}

pub(super) fn contains_assignment_operator(text: &str) -> bool {
    let chars = text.chars().collect::<Vec<_>>();
    for (idx, ch) in chars.iter().enumerate() {
        if *ch != '=' {
            continue;
        }
        let prev = idx.checked_sub(1).and_then(|left| chars.get(left)).copied();
        let next = chars.get(idx + 1).copied();
        if matches!(prev, Some('=' | '!' | '<' | '>')) || next == Some('=') {
            continue;
        }
        return true;
    }
    false
}

pub(super) fn contains_top_level_for_in_tail(text: &str) -> bool {
    let chars = text.chars().collect::<Vec<_>>();
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    let mut saw_for = false;
    let mut idx = 0usize;
    while idx < chars.len() {
        let ch = chars[idx];
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == active_quote {
                quote = None;
            }
            idx += 1;
            continue;
        }
        match ch {
            '"' | '\'' => {
                quote = Some(ch);
                idx += 1;
                continue;
            }
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = (depth - 1).max(0),
            _ => {}
        }
        if depth == 0 && word_at(&chars, idx, "for") {
            saw_for = true;
            idx += 3;
            continue;
        }
        if saw_for && depth == 0 && word_at(&chars, idx, "in") {
            return true;
        }
        idx += 1;
    }
    false
}

pub(super) fn word_at(chars: &[char], idx: usize, word: &str) -> bool {
    let word_chars = word.chars().collect::<Vec<_>>();
    if idx + word_chars.len() > chars.len() {
        return false;
    }
    for (offset, expected) in word_chars.iter().enumerate() {
        if !chars[idx + offset].eq_ignore_ascii_case(expected) {
            return false;
        }
    }
    let before_ok = idx == 0 || !is_identifier_char(chars[idx - 1]);
    let after_idx = idx + word_chars.len();
    let after_ok = after_idx >= chars.len() || !is_identifier_char(chars[after_idx]);
    before_ok && after_ok
}

pub(super) fn is_identifier_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

pub(super) fn useful_body(body: &str) -> bool {
    let trimmed = body.trim();
    if trimmed.is_empty() || trimmed.len() > 900 {
        return false;
    }
    let lowered = trimmed.to_lowercase();
    let blocked = [
        "open(",
        "exec(",
        "eval(",
        "__",
        "import ",
        "assert ",
        "sys.",
        "subprocess",
        "os.",
        "task_id",
        "canonical",
        "solution",
        "candidate generated",
        "private curriculum",
    ];
    if blocked.iter().any(|token| lowered.contains(token)) {
        return false;
    }
    if !lowered.contains("return") || !trimmed.chars().any(|ch| ch.is_ascii_alphanumeric()) {
        return false;
    }
    delimiters_balanced(trimmed)
}

pub(super) fn useful_generated_body_for_task(task: &CodeTask, body: &str) -> bool {
    if useful_generated_body(body) {
        return true;
    }
    useful_task_scoped_system_body(task, body)
}

pub(super) fn useful_task_scoped_system_body(task: &CodeTask, body: &str) -> bool {
    let trimmed = body.trim();
    let max_len = if execution_shaped_category(&task.category) {
        4000
    } else {
        1200
    };
    if trimmed.is_empty() || trimmed.len() > max_len {
        return false;
    }
    if !trimmed.chars().any(|ch| ch.is_ascii_alphanumeric()) {
        return false;
    }
    let lowered = trimmed.to_lowercase();
    if !lowered.contains("return") {
        return false;
    }
    let hard_blocked = [
        "exec(",
        "eval(",
        "__",
        "assert ",
        "sys.",
        "task_id",
        "canonical",
        "solution",
        "candidate generated",
        "private curriculum",
    ];
    if hard_blocked.iter().any(|token| lowered.contains(token)) {
        return false;
    }
    let task_text = format!("{} {}", task.category, task.prompt).to_lowercase();
    let file_prompt = task_text.contains("file")
        || task_text.contains("directory")
        || task_text.contains("folder")
        || task_text.contains("path")
        || task_text.contains("archive")
        || task_text.contains("zip")
        || task_text.contains("csv")
        || task_text.contains(".log")
        || task_text.contains("backup")
        || task_text.contains("read")
        || task_text.contains("write")
        || task_text.contains("output");
    let process_prompt = task_text.contains("process")
        && (task_text.contains("running")
            || task_text.contains("restart")
            || task_text.contains("terminate")
            || task_text.contains("command"));
    let command_prompt = task_text.contains("shell command")
        || task_text.contains("commands")
        || task_text.contains("command")
        || process_prompt;
    let system_info_prompt = task_text.contains("operating system")
        || task_text.contains("architecture")
        || task_text.contains("memory usage")
        || task_text.contains("system information")
        || process_prompt;
    let crypto_prompt = task_text.contains("pbkdf2")
        || task_text.contains("hashlib")
        || task_text.contains("os.urandom")
        || (task_text.contains("password") && task_text.contains("salt"))
        || (task_text.contains("hash") && task_text.contains("salt"));
    let math_prompt = task_text.contains("gcd")
        || task_text.contains("coprime")
        || task_text.contains("prime")
        || task_text.contains("divisor");
    if lowered.contains("open(") && !file_prompt {
        return false;
    }
    if (lowered.contains("os.path") || lowered.contains("pathlib.")) && !file_prompt {
        return false;
    }
    if lowered.contains("os.") && !(file_prompt || crypto_prompt) {
        return false;
    }
    if lowered.contains("subprocess") && !command_prompt {
        return false;
    }
    if lowered.contains("platform.") && !system_info_prompt {
        return false;
    }
    if lowered.contains("psutil") && !system_info_prompt {
        return false;
    }
    if lowered.contains("import ")
        && !body_imports_are_task_scoped_safe(
            body,
            &task_text,
            file_prompt,
            command_prompt,
            system_info_prompt,
            crypto_prompt,
            math_prompt,
        )
    {
        return false;
    }
    delimiters_balanced(trimmed)
}

pub(super) fn body_imports_are_task_scoped_safe(
    body: &str,
    task_text: &str,
    file_prompt: bool,
    command_prompt: bool,
    system_info_prompt: bool,
    crypto_prompt: bool,
    math_prompt: bool,
) -> bool {
    for raw_line in body.lines() {
        let line = raw_line.trim();
        if line.starts_with("import ") {
            for raw in line.trim_start_matches("import ").split(',') {
                let item = raw.trim();
                if item.is_empty() {
                    continue;
                }
                let base = item
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .split('.')
                    .next()
                    .unwrap_or("");
                if !safe_task_body_import_module(
                    base,
                    task_text,
                    file_prompt,
                    command_prompt,
                    system_info_prompt,
                    crypto_prompt,
                    math_prompt,
                ) {
                    return false;
                }
            }
        } else if line.starts_with("from ") {
            let Some((module_part, names_part)) =
                line.trim_start_matches("from ").split_once(" import ")
            else {
                return false;
            };
            if names_part.contains('*') {
                return false;
            }
            let base = module_part.trim().split('.').next().unwrap_or("");
            if !safe_task_body_import_module(
                base,
                task_text,
                file_prompt,
                command_prompt,
                system_info_prompt,
                crypto_prompt,
                math_prompt,
            ) {
                return false;
            }
        }
    }
    true
}

pub(super) fn safe_task_body_import_module(
    module: &str,
    task_text: &str,
    file_prompt: bool,
    command_prompt: bool,
    system_info_prompt: bool,
    crypto_prompt: bool,
    math_prompt: bool,
) -> bool {
    match module {
        "csv" | "glob" | "gzip" | "pathlib" | "shutil" | "tarfile" | "tempfile" | "zipfile"
        | "configparser" => file_prompt,
        "os" => file_prompt || crypto_prompt,
        "subprocess" => command_prompt,
        "platform" | "psutil" => system_info_prompt || task_text.contains(module),
        "math" => math_prompt || safe_visible_import_module(module),
        _ => safe_visible_import_module(module),
    }
}

pub(super) fn useful_generated_body(body: &str) -> bool {
    if !useful_body(body) {
        return false;
    }
    let trimmed = body.trim();
    if trimmed == "return" || trimmed == "return None" {
        return false;
    }
    for line in trimmed.lines() {
        let line = line.trim();
        if line.ends_with(|ch: char| "+-*/%<>=!&|.,".contains(ch)) {
            return false;
        }
        if line.starts_with("return") && line != "return" && !line.starts_with("return ") {
            return false;
        }
        if let Some(expr) = line.strip_prefix("return ") {
            if !useful_generated_expression(expr) {
                return false;
            }
        }
        if line.starts_with("return =") || line.starts_with("return :") || line.contains(".0") {
            return false;
        }
        if line.starts_with("return ") && line.ends_with(':') {
            return false;
        }
        if invalid_overcomposed_generated_line(line) {
            return false;
        }
        if invalid_partial_member_line(line) {
            return false;
        }
        if invalid_inline_block_header_body(line) {
            return false;
        }
        if (line.starts_with("if ")
            || line.starts_with("for ")
            || line.starts_with("while ")
            || line.starts_with("with ")
            || line.starts_with("try")
            || line.starts_with("except ")
            || line.starts_with("elif ")
            || line == "else")
            && !line.ends_with(':')
        {
            return false;
        }
    }
    true
}

pub(super) fn syntax_constrained_body(body: &str) -> bool {
    if !python_try_blocks_have_handlers(body) {
        return false;
    }
    let mut saw_return = false;
    let mut previous_indent = 0usize;
    let mut previous_line_opened_block = false;
    let mut saw_top_level_return = false;
    let mut assigned_names = BTreeSet::new();
    for raw_line in body.lines() {
        if raw_line.trim().is_empty() {
            continue;
        }
        let indent = raw_line.chars().take_while(|ch| *ch == ' ').count() / 4;
        let line = raw_line.trim();
        if previous_line_opened_block && indent <= previous_indent {
            return false;
        }
        if !previous_line_opened_block && indent > previous_indent {
            return false;
        }
        if matches!(line, "if:" | "for:" | "while:" | "elif:" | "else") {
            return false;
        }
        if line.contains("lambda ") && line.contains(" as ") {
            return false;
        }
        if invalid_lambda_syntax(line) {
            return false;
        }
        if invalid_block_header(line) {
            return false;
        }
        if line.starts_with("return ") {
            saw_return = true;
            if indent == 0 {
                saw_top_level_return = true;
            }
        }
        if line.starts_with("return") && line != "return" && !line.starts_with("return ") {
            return false;
        }
        if line == "return"
            || line.contains("return return")
            || line.starts_with("return =")
            || line.starts_with("return :")
        {
            return false;
        }
        if line == "raise" {
            return false;
        }
        if let Some(expr) = line.strip_prefix("return ") {
            if !useful_generated_expression(expr) {
                return false;
            }
            if matches!(
                expr.trim(),
                "out" | "total" | "best" | "numbers" | "values" | "stack"
            ) && !assigned_names.contains(expr.trim())
            {
                return false;
            }
            if matches!(expr.trim(), "nondecreasing" | "nonincreasing")
                && !assigned_names.contains(expr.trim())
            {
                return false;
            }
        }
        if line.starts_with("return ") && line.ends_with(':') {
            return false;
        }
        if invalid_overcomposed_generated_line(line) {
            return false;
        }
        if invalid_partial_member_line(line) {
            return false;
        }
        if invalid_inline_block_header_body(line) {
            return false;
        }
        if (line.starts_with("if ")
            || line.starts_with("for ")
            || line.starts_with("while ")
            || line.starts_with("with ")
            || line.starts_with("try")
            || line.starts_with("except ")
            || line.starts_with("elif ")
            || line == "else")
            && !line.ends_with(':')
        {
            return false;
        }
        if let Some((name, _)) = line.split_once('=') {
            let name = name.trim();
            if is_identifier(name) {
                assigned_names.insert(name.to_string());
            }
        }
        previous_indent = indent;
        previous_line_opened_block = line.ends_with(':');
    }
    saw_return && saw_top_level_return
}

pub(super) fn transformer_hybrid_import_body_ok(body: &str) -> bool {
    if syntax_constrained_body(body) {
        return true;
    }
    if !python_try_blocks_have_handlers(body) {
        return false;
    }
    let mut saw_return = false;
    let mut previous_indent = 0usize;
    let mut previous_line_opened_block = false;
    let mut non_empty_lines = 0usize;
    for raw_line in body.lines() {
        if raw_line.trim().is_empty() {
            continue;
        }
        non_empty_lines += 1;
        let indent = raw_line.chars().take_while(|ch| *ch == ' ').count() / 4;
        let line = raw_line.trim();
        if previous_line_opened_block && indent <= previous_indent {
            return false;
        }
        if !previous_line_opened_block && indent > previous_indent {
            return false;
        }
        if line.contains("__")
            || line.contains("eval(")
            || line.contains("exec(")
            || line.contains("open(")
            || line.contains("subprocess")
            || line.contains("os.")
            || line.contains("sys.")
            || line.contains("socket")
            || line.contains("requests")
            || line.contains("urllib")
            || line.contains("pickle")
        {
            return false;
        }
        if line.starts_with("import ") || line.starts_with("from ") {
            if !transformer_hybrid_safe_import_line(line) {
                return false;
            }
            previous_indent = indent;
            previous_line_opened_block = false;
            continue;
        }
        if matches!(line, "if:" | "for:" | "while:" | "elif:" | "else") {
            return false;
        }
        if line.starts_with("return ") || line == "return" {
            saw_return = true;
        }
        if line == "pass"
            || line == "..."
            || line.contains("TODO")
            || line.contains("NotImplemented")
            || line.contains("return return")
            || line.starts_with("return =")
            || line.starts_with("return :")
        {
            return false;
        }
        if line.starts_with("return") && line != "return" && !line.starts_with("return ") {
            return false;
        }
        if invalid_lambda_syntax(line) || invalid_block_header(line) {
            return false;
        }
        if (line.starts_with("def ")
            || line.starts_with("if ")
            || line.starts_with("for ")
            || line.starts_with("while ")
            || line.starts_with("with ")
            || line.starts_with("try")
            || line.starts_with("except ")
            || line.starts_with("elif ")
            || line == "else")
            && !line.ends_with(':')
        {
            return false;
        }
        previous_indent = indent;
        previous_line_opened_block = line.ends_with(':');
    }
    saw_return && non_empty_lines > 0
}

fn transformer_hybrid_safe_import_line(line: &str) -> bool {
    matches!(
        line,
        "import functools"
            | "import itertools"
            | "import math"
            | "import re"
            | "from collections import Counter"
            | "from typing import *"
    )
}

pub(super) fn python_try_blocks_have_handlers(body: &str) -> bool {
    let lines = body
        .lines()
        .filter_map(|raw| {
            let line = raw.trim();
            if line.is_empty() {
                None
            } else {
                Some((raw.chars().take_while(|ch| *ch == ' ').count() / 4, line))
            }
        })
        .collect::<Vec<_>>();
    for (idx, (try_indent, line)) in lines.iter().enumerate() {
        if *line != "try:" && *line != "try" {
            continue;
        }
        let mut found_indented_body = false;
        let mut found_handler = false;
        for (indent, candidate) in lines.iter().skip(idx + 1) {
            if *indent > *try_indent {
                found_indented_body = true;
                continue;
            }
            if *indent == *try_indent
                && (candidate.starts_with("except") || candidate.starts_with("finally:"))
            {
                found_handler = found_indented_body;
                break;
            }
            if *indent <= *try_indent {
                break;
            }
        }
        if !found_handler {
            return false;
        }
    }
    true
}

pub(super) fn invalid_block_header(line: &str) -> bool {
    let header = line.trim();
    if invalid_inline_block_header_body(header) {
        return true;
    }
    if header.starts_with("if ") && archive_context_manager_header_text(header) {
        return true;
    }
    if header.starts_with("with ") && invalid_context_manager_header_text(header) {
        return true;
    }
    if malformed_adjacent_call_header_text(header) {
        return true;
    }
    for keyword in ["if", "elif", "while"] {
        if let Some(rest) = header.strip_prefix(&format!("{keyword} ")) {
            let expr = rest.trim_end_matches(':').trim();
            if expr.is_empty()
                || expr.starts_with('=')
                || expr.starts_with(':')
                || expr.starts_with(',')
                || expr.starts_with(')')
                || expr.starts_with(']')
                || conditional_expr_contains_bare_assignment(expr)
            {
                return true;
            }
        }
    }
    if let Some(rest) = header.strip_prefix("for ") {
        let expr = rest.trim_end_matches(':').trim();
        if expr.is_empty() || expr.starts_with("in ") || !expr.contains(" in ") {
            return true;
        }
    }
    false
}

pub(super) fn malformed_adjacent_call_header_text(header: &str) -> bool {
    let lowered = header.to_lowercase();
    [
        ") configparser.",
        ") configparser(",
        ") os.",
        ") json.",
        ") csv.",
        ") zipfile.",
        ") tarfile.",
        ") shutil.",
        ") filenotfounderror",
        ") true",
        ") false",
    ]
    .iter()
    .any(|needle| lowered.contains(needle))
}

pub(super) fn conditional_expr_contains_bare_assignment(expr: &str) -> bool {
    let chars = expr.chars().collect::<Vec<_>>();
    for (idx, ch) in chars.iter().enumerate() {
        if *ch != '=' {
            continue;
        }
        let prev = idx.checked_sub(1).and_then(|pos| chars.get(pos)).copied();
        let next = chars.get(idx + 1).copied();
        if matches!(prev, Some('=' | '!' | '<' | '>' | ':')) || matches!(next, Some('=')) {
            continue;
        }
        return true;
    }
    false
}

pub(super) fn archive_context_manager_header_text(header: &str) -> bool {
    let lowered = header.to_lowercase();
    lowered.contains("tarfile.open") || lowered.contains("zipfile.zipfile")
}

pub(super) fn invalid_context_manager_header_text(header: &str) -> bool {
    let lowered = header.to_lowercase();
    lowered.contains("os.path.isfile")
        || lowered.contains("os.path.isdir")
        || lowered.contains("os.path.exists")
        || lowered.contains("csv.reader")
        || lowered.contains("list(")
}

pub(super) fn invalid_overcomposed_generated_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.len() > 240 {
        return true;
    }
    let lowered = trimmed.to_lowercase();
    if lowered.contains(".raise ") || lowered.contains(" = raise ") {
        return true;
    }
    if invalid_bare_adjacent_identifier_call_line(trimmed) {
        return true;
    }
    if [
        "exist_ok",
        "capture_output",
        "text = true",
        "encoding =",
        "newline =",
    ]
    .iter()
    .any(|needle| lowered.matches(needle).count() > 1)
    {
        return true;
    }
    if lowered.contains("with open")
        && (lowered.contains("row in enumerate") || lowered.contains("subprocess.run"))
    {
        return true;
    }
    if lowered.contains(".join (") && lowered.contains("exist_ok") {
        return true;
    }
    lowered.matches(" = ").count() > 4
}

pub(super) fn invalid_bare_adjacent_identifier_call_line(line: &str) -> bool {
    let tokens = tokenize_code(line);
    if tokens.len() < 3 {
        return false;
    }
    if !is_identifier(&tokens[0]) || !is_identifier(&tokens[1]) || tokens[2] != "(" {
        return false;
    }
    !matches!(
        tokens[0].as_str(),
        "return"
            | "raise"
            | "import"
            | "from"
            | "if"
            | "for"
            | "while"
            | "with"
            | "try"
            | "except"
            | "elif"
            | "else"
    )
}

pub(super) fn invalid_partial_member_line(line: &str) -> bool {
    let compact = line
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    compact.contains("os.path.]")
        || compact.contains(".path.]")
        || compact.contains(".join:)")
        || compact.contains(".join:]")
        || compact.contains("os.path.join]")
        || compact.contains("os.path.join:)")
        || compact.ends_with("os.path.")
        || compact.ends_with(".")
}

pub(super) fn invalid_inline_block_header_body(line: &str) -> bool {
    let header = line.trim();
    if !starts_python_block_header(header) {
        return false;
    }
    let Some(idx) = first_top_level_colon_index(header) else {
        return false;
    };
    !header[idx + 1..].trim().is_empty()
}

pub(super) fn starts_python_block_header(header: &str) -> bool {
    header.starts_with("if ")
        || header.starts_with("for ")
        || header.starts_with("while ")
        || header.starts_with("with ")
        || header.starts_with("except ")
        || header.starts_with("elif ")
        || header == "else"
        || header == "else:"
        || header.starts_with("else:")
        || header == "try"
        || header == "try:"
        || header.starts_with("try:")
}

pub(super) fn first_top_level_colon_index(text: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (idx, ch) in text.char_indices() {
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == active_quote {
                quote = None;
            }
            continue;
        }
        match ch {
            '"' | '\'' => quote = Some(ch),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = (depth - 1).max(0),
            ':' if depth == 0 => return Some(idx),
            _ => {}
        }
    }
    None
}

pub(super) fn invalid_lambda_syntax(line: &str) -> bool {
    let Some(lambda_pos) = line.find("lambda ") else {
        return false;
    };
    let after = &line[lambda_pos + "lambda ".len()..];
    let Some(colon_pos) = after.find(':') else {
        return true;
    };
    let params = after[..colon_pos].trim();
    if params.is_empty() {
        return true;
    }
    params
        .split(',')
        .map(str::trim)
        .any(|param| !is_identifier(param))
}

pub(super) fn extract_first_return_expression(body: &str) -> Option<String> {
    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(expr) = trimmed.strip_prefix("return ") {
            let expr = expr.trim();
            if useful_generated_expression(expr) {
                return Some(expr.to_string());
            }
        }
    }
    None
}

pub(super) fn delimiters_balanced(text: &str) -> bool {
    let mut stack = Vec::new();
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for ch in text.chars() {
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == active_quote {
                quote = None;
            }
            continue;
        }
        match ch {
            '"' | '\'' => quote = Some(ch),
            '(' | '[' | '{' => stack.push(ch),
            ')' => {
                if stack.pop() != Some('(') {
                    return false;
                }
            }
            ']' => {
                if stack.pop() != Some('[') {
                    return false;
                }
            }
            '}' => {
                if stack.pop() != Some('{') {
                    return false;
                }
            }
            _ => {}
        }
    }
    quote.is_none() && stack.is_empty()
}

pub(super) fn is_identifier(token: &str) -> bool {
    let mut chars = token.chars();
    match chars.next() {
        Some(ch) if ch.is_ascii_alphabetic() || ch == '_' => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

pub(super) fn sanitize_ident(value: &str) -> String {
    let mut out = String::new();
    for (idx, ch) in value.chars().enumerate() {
        if (idx == 0 && (ch.is_ascii_alphabetic() || ch == '_'))
            || (idx > 0 && (ch.is_ascii_alphanumeric() || ch == '_'))
        {
            out.push(ch);
        }
    }
    if out.is_empty() {
        "solve".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn minimal_task(task_id: &str) -> CodeTask {
        CodeTask {
            raw: json!({
                "task_id": task_id,
                "prompt": "Return the input.",
                "entry_point": "solve"
            }),
            task_id: task_id.to_string(),
            source_task_id: task_id.to_string(),
            card_id: "private_test_card".to_string(),
            source_id: "private_test_source".to_string(),
            split: "heldout".to_string(),
            category: "unit".to_string(),
            prompt: "Return the input.".to_string(),
            entry_point: "solve".to_string(),
            solution_expr: "data".to_string(),
            solution_body: "return data".to_string(),
            tags: Vec::new(),
            benchmark_evidence_level: "private_generated_only".to_string(),
        }
    }

    #[test]
    fn sts_control_policy_does_not_create_streams_for_control_lane() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "theseus-sts-control-policy-{}-{}.jsonl",
            std::process::id(),
            "no-create"
        ));
        std::fs::write(
            &path,
            "{\"objective\":\"decode visible contract\",\"answer\":\"respect private metadata\"}\n",
        )
        .unwrap();

        let tasks = vec![minimal_task("private-task-1")];
        let mut empty_streams = StsStreamMap::new();
        let applied =
            merge_sts_decoder_control_policy_for_tasks(&mut empty_streams, &tasks, &[], &path)
                .unwrap();

        assert_eq!(applied, 0);
        assert!(empty_streams.is_empty());

        let mut seeded_streams = StsStreamMap::new();
        seeded_streams.insert("private-task-1".to_string(), BTreeMap::new());
        let applied =
            merge_sts_decoder_control_policy_for_tasks(&mut seeded_streams, &tasks, &[], &path)
                .unwrap();

        assert_eq!(applied, 1);
        assert!(seeded_streams["private-task-1"].contains_key("decoder_control_stream"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn inferred_prompt_preserves_four_arg_contract() {
        let raw = json!({
            "entry_point": "solve",
            "prompt": "Return shortest hop count.",
            "decoder_contract": {"visible_arg_count_hint": 4}
        });

        assert_eq!(
            prompt_with_inferred_signature("solve", "Return shortest hop count.", &raw),
            "def solve(data, other, third, fourth):\nReturn shortest hop count."
        );
    }

    #[test]
    fn inferred_arg_count_from_tests_can_exceed_three() {
        let raw = json!({
            "entry_point": "solve",
            "tests": "assert solve(5, [(0, 1)], 0, 2) == 2\n"
        });

        assert_eq!(inferred_arg_count_from_tests(&raw), Some(4));
        assert_eq!(inferred_visible_arg_count(&raw), 4);
    }

    #[test]
    fn canonicalize_preserves_extra_tuple_rest_alias() {
        let signature = VisibleSignature {
            header: "def solve(data, other, third, fourth):".to_string(),
            arg_names: vec![
                "data".to_string(),
                "other".to_string(),
                "third".to_string(),
                "fourth".to_string(),
            ],
            uses_varargs: false,
        };

        let body = "start = extra[0] if len(extra) > 0 else 0";
        assert_eq!(canonicalize_candidate_body_aliases(body, &signature), body);
        assert!(body_needs_signature_aliases(body));
    }
}
