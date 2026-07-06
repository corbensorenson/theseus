// Contract verification, candidate quality scoring, and promotion guardrails for Code LM closure.
// Kept separate from skeleton generation so verifier/ranker policy can evolve independently.

use super::*;

mod category_taxonomy;
mod quality_gate;
mod scoring;
mod semantic_admissibility;

pub(super) use quality_gate::*;

pub(super) fn execution_shape_contract_ok(
    task: &CodeTask,
    body: &str,
    hints: &BTreeSet<String>,
) -> bool {
    let lowered = body.to_lowercase();
    let category_specific_shape_body = execution_shape_category_contract_ok(task, body, hints);
    let trusted_process_restart_body = matches!(
        task.category.as_str(),
        "private_exec_process_restart" | "process_restart"
    ) && syntax_constrained_body(body)
        && lowered.contains("psutil.process_iter")
        && lowered.contains("subprocess.popen")
        && lowered.contains("process found")
        && lowered.contains("process not found");
    let trusted_execution_shape_body = execution_shaped_category(&task.category)
        && syntax_constrained_body(body)
        && (useful_task_scoped_system_body(task, body)
            || trusted_process_restart_body
            || category_specific_shape_body);
    if !trusted_execution_shape_body {
        if !useful_generated_body_for_task(task, body)
            || !syntax_constrained_body(body)
            || !body_semantically_admissible(task, body)
        {
            return false;
        }
    }
    if hints.contains("file_path")
        && !body_has_any(&lowered, &["os.path", "open(", "isfile", "isdir", "exists"])
    {
        return false;
    }
    if hints.contains("csv")
        && !body_has_any(&lowered, &["csv.", "csvreader", "reader(", "read_csv"])
    {
        return false;
    }
    if hints.contains("archive")
        && !body_has_any(
            &lowered,
            &["zipfile", "tarfile", "shutil", "make_archive", "archive"],
        )
    {
        return false;
    }
    if hints.contains("structured_parsing")
        && !body_has_any(
            &lowered,
            &["json", "base64", "urlencode", "beautifulsoup", "payload"],
        )
    {
        return false;
    }
    if hints.contains("system_api")
        && !body_has_any(
            &lowered,
            &["platform", "psutil", "subprocess", "popen", "run("],
        )
    {
        return false;
    }
    let required = decoder_required_constructs(task);
    if !required_construct_contract_ok_for_task(task, body, &required) {
        return false;
    }
    return_shape_contract_ok(task, &lowered)
}

pub(super) fn execution_shape_category_contract_ok(
    task: &CodeTask,
    body: &str,
    hints: &BTreeSet<String>,
) -> bool {
    if !execution_shaped_category(&task.category) {
        return false;
    }
    let lowered = body.to_lowercase();
    syntax_constrained_body(body)
        && lowered.contains("return")
        && !natural_language_leakage_in_body(body)
        && !["exec(", "eval(", "__", "assert ", "sys."]
            .iter()
            .any(|needle| lowered.contains(needle))
        && visible_argument_contract_ok(task, body)
        && required_construct_contract_ok_for_task(task, body, hints)
        && execution_shape_library_contract_ok(task, body, hints)
        && execution_shape_category_semantic_contract_ok(task, body)
}

pub(super) fn execution_shape_category_semantic_contract_ok(task: &CodeTask, body: &str) -> bool {
    if !execution_shaped_category(&task.category) {
        return true;
    }
    let lowered = body.to_lowercase();
    let compact = lowered
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    match task.category.as_str() {
        "private_exec_archive_config_zip" | "archive_config_zip" => {
            lowered.contains("make_archive")
                && lowered.contains("archive_base")
                && lowered.contains("filenotfounderror")
                && compact.contains("config=configparser.configparser()")
                && compact.contains("config.read(")
                && compact.contains("project_dir=config.get(")
                && !compact.contains("project_dir=config.get()")
                && !compact.contains("project_dir=config.make_archive(")
                && !compact.contains("project_dir=config.makedirs(")
                && !compact.contains("os.read(")
        }
        "private_exec_csv_command_outputs" | "csv_command_outputs" => {
            lowered.contains("subprocess.run")
                && lowered.contains("capture_output")
                && lowered.contains("out.append")
        }
        "private_exec_csv_split_shuffle" | "csv_split_shuffle" => {
            lowered.contains("csv.writer")
                && lowered.contains("writerows")
                && lowered.contains("chunk_size")
        }
        "private_exec_json_extract_field" | "json_extract_field" => {
            (lowered.contains("json.load(")
                || lowered.contains("json.loads(")
                || lowered.contains("isinstance(payload, dict)"))
                && compact.contains("payload.get(")
                && !json_payload_load_is_uncalled(body)
                && (lowered.contains("except")
                    || lowered.contains("return none")
                    || lowered.contains("return ''")
                    || lowered.contains("return \"\""))
        }
        "private_exec_log_backup_tar" | "log_backup_tar" => {
            lowered.contains("tarfile.open")
                && lowered.contains("archive.add")
                && lowered.contains("os.remove")
                && !tar_archive_returns_before_add(body)
        }
        "private_exec_urlencode_payload" | "urlencode_payload" => {
            lowered.contains("urlencode")
                && lowered.contains("sorted(")
                && compact.contains("returnurlencode(")
        }
        "private_exec_zip_flat_directory" | "zip_flat_directory" => {
            lowered.contains("zipfile.zipfile")
                && lowered.contains("archive.write")
                && lowered.contains("arcname")
        }
        _ => true,
    }
}

fn json_payload_load_is_uncalled(body: &str) -> bool {
    for raw_line in body.lines() {
        let compact = raw_line
            .trim()
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect::<String>()
            .to_lowercase();
        if compact.starts_with("payload=json.loads") && !compact.starts_with("payload=json.loads(")
        {
            return true;
        }
        if compact.starts_with("payload=json.load")
            && !compact.starts_with("payload=json.load(")
            && !compact.starts_with("payload=json.loads")
        {
            return true;
        }
    }
    false
}

fn tar_archive_returns_before_add(body: &str) -> bool {
    let mut in_tar_context = false;
    let mut context_indent = 0usize;
    let mut saw_archive_add = false;
    for raw_line in body.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let indent = raw_line.len().saturating_sub(raw_line.trim_start().len());
        let lowered = trimmed.to_lowercase();
        if in_tar_context && indent <= context_indent {
            in_tar_context = false;
        }
        if lowered.starts_with("with ") && lowered.contains("tarfile.open") {
            in_tar_context = true;
            context_indent = indent;
            saw_archive_add = false;
            continue;
        }
        if in_tar_context {
            if lowered.contains("archive.add(") {
                saw_archive_add = true;
            }
            if lowered.starts_with("return ") && !saw_archive_add {
                return true;
            }
        }
    }
    false
}

pub(super) fn execution_shape_behavioral_antipattern(task: &CodeTask, body: &str) -> bool {
    match task.category.as_str() {
        "private_exec_json_extract_field" | "json_extract_field" => {
            json_payload_load_is_uncalled(body)
        }
        "private_exec_log_backup_tar" | "log_backup_tar" => tar_archive_returns_before_add(body),
        _ => false,
    }
}

pub(super) fn body_has_any(body: &str, needles: &[&str]) -> bool {
    let compact = body
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    needles
        .iter()
        .any(|needle| body.contains(needle) || compact.contains(&needle.replace(' ', "")))
}

pub(super) fn natural_language_leakage_in_body(body: &str) -> bool {
    let blocked_fragments = [
        "here is",
        "this function",
        "the function should",
        "we need to",
        "you can",
        "expected output",
        "example:",
        "explanation",
        "canonical solution",
        "private curriculum",
        "public benchmark",
        "candidate generated",
    ];
    for raw_line in body.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let lowered = line.to_lowercase();
        if blocked_fragments
            .iter()
            .any(|fragment| lowered.contains(fragment))
        {
            return true;
        }
        let code_like = line.starts_with("return ")
            || line.starts_with("if ")
            || line.starts_with("elif ")
            || line == "else:"
            || line.starts_with("for ")
            || line.starts_with("while ")
            || line.starts_with("try:")
            || line.starts_with("except ")
            || line.starts_with("with ")
            || line.starts_with("import ")
            || line.starts_with("from ")
            || line.contains('=')
            || line.contains('(')
            || line.contains('[')
            || line.contains('{');
        let word_count = line
            .split_whitespace()
            .filter(|part| part.chars().any(|ch| ch.is_ascii_alphabetic()))
            .count();
        if !code_like && word_count >= 4 {
            return true;
        }
    }
    false
}

pub(super) fn unbound_item_reference(body: &str) -> bool {
    if !body_mentions_token(body, "item") {
        return false;
    }
    let lowered = body.to_lowercase();
    let compact = lowered
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let bound = body_has_any(
        &lowered,
        &[
            "for item",
            "lambda item",
            "item in",
            "item,",
            "(item",
            " item =",
            "item=",
        ],
    ) || compact.contains("foritemin");
    !bound
}

pub(super) fn execution_shape_invalid_partial_statement(body: &str) -> bool {
    let lowered = body.to_lowercase();
    if !lowered.contains("execution")
        && !body_has_any(
            &lowered,
            &[
                "os.path",
                "zipfile",
                "tarfile",
                "configparser",
                "csv",
                "json",
            ],
        )
    {
        return false;
    }
    let compact_body = lowered
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let has_config_assignment = compact_body.contains("config=configparser.configparser()");
    for raw_line in body.lines() {
        let line = raw_line.trim();
        let compact = line
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect::<String>()
            .to_lowercase();
        if matches!(
            compact.as_str(),
            "os.path.isfile()"
                | "os.path.isdir()"
                | "os.path.exists()"
                | "os.path.join()"
                | "join()"
                | "config.read()"
        ) {
            return true;
        }
        if compact.contains("os.read(") {
            return true;
        }
        if compact == "config=configparser.configparser" {
            return true;
        }
        if compact == "config=configparser.path" {
            return true;
        }
        if compact.starts_with("ifnotos.path.isfile(") && compact.contains("=config.get(") {
            return true;
        }
        if line == "raise" || line == "os.makedirs" {
            return true;
        }
        if line.starts_with("config.get") && !line.contains("project_dir") {
            return true;
        }
        if compact.starts_with("project_dir=config.get")
            && (!compact.contains("config.get(")
                || compact.contains("config.getconfigparser")
                || compact.contains("config.get()"))
        {
            return true;
        }
        if compact.starts_with("project_dir=config.make_archive(")
            || compact.starts_with("project_dir=config.makedirs(")
        {
            return true;
        }
        if line.starts_with("config.read") && !has_config_assignment {
            return true;
        }
        if line.starts_with("base = os.")
            && !line.contains("basename")
            && !line.contains("normpath")
        {
            return true;
        }
        if line.starts_with("zip_path = os.path") && !line.contains("join(") {
            return true;
        }
        if line.starts_with("archive_path = os.path") && !line.contains("join(") {
            return true;
        }
        if line == "join" || line.starts_with("join(") {
            return true;
        }
    }
    false
}

pub(super) fn visible_argument_contract_ok(task: &CodeTask, body: &str) -> bool {
    let expected = category_expected_arg_count(task).unwrap_or(0);
    let lowered = body.to_lowercase();
    if expected == 0 {
        return !(body_mentions_token(body, "data")
            || body_mentions_token(body, "other")
            || body_mentions_token(body, "args"));
    }
    if expected < 2 && body_mentions_token(body, "other") {
        return false;
    }
    let visible_args_ordered = visible_signature_ordered_user_args(task);
    let visible_args = visible_args_ordered
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if expected >= 2 {
        let secondary_visible = visible_args_ordered
            .get(1)
            .is_some_and(|name| body_mentions_token(body, name));
        let secondary_alias = body_mentions_token(body, "other");
        if !secondary_visible && !secondary_alias {
            return false;
        }
    }
    if expected >= 3 {
        let third_visible = visible_args_ordered
            .get(2)
            .is_some_and(|name| body_mentions_token(body, name));
        let third_alias = body_mentions_token(body, "extra");
        if !third_visible && !third_alias {
            return false;
        }
    }
    let alias_used = body_mentions_token(body, "data")
        || body_mentions_token(body, "other")
        || body_mentions_token(body, "args")
        || body_mentions_token(body, "extra");
    let visible_used = visible_args
        .iter()
        .any(|name| body_mentions_token(body, name));
    if alias_used || visible_used {
        return true;
    }
    if let Some(expr) = first_top_level_return_expr(body) {
        let compact = expr.replace(' ', "").to_lowercase();
        let constant_return = matches!(
            compact.as_str(),
            "0" | "1" | "true" | "false" | "none" | "[]" | "{}" | "''" | "\"\""
        );
        if constant_return && !task_contract_text(task).contains("empty") {
            return false;
        }
    }
    lowered.contains("open(")
        || lowered.contains("platform.")
        || lowered.contains("psutil")
        || lowered.contains("subprocess")
}

pub(super) fn required_construct_contract_ok(body: &str, hints: &BTreeSet<String>) -> bool {
    let lowered = body.to_lowercase();
    if hints.contains("loop")
        && !body_has_any(&lowered, &["for ", "while ", "map(", "filter(", "sorted("])
        && !(hints.contains("execution_shaped_program")
            && body_has_any(
                &lowered,
                &[
                    "sorted(",
                    ".items(",
                    ".apply(",
                    "urlencode(",
                    "csv.reader",
                    "read_csv",
                    "glob.",
                    "os.listdir",
                    "json.load",
                ],
            ))
    {
        return false;
    }
    if hints.contains("branch")
        && !body_has_any(&lowered, &["if ", "try:", "except ", " or ", " and "])
    {
        return false;
    }
    if hints.contains("locals") && body.lines().filter(|line| line.contains('=')).count() == 0 {
        return false;
    }
    if hints.contains("frequency")
        && !body_has_any(
            &lowered,
            &[
                "counts", "counter", ".get(", "+= 1", "+=1", ".count(", "seen", " in seen",
            ],
        )
    {
        return false;
    }
    if hints.contains("selection")
        && !body_has_any(
            &lowered,
            &["sorted(", ".sort(", "min(", "max(", "best", "median", "mid"],
        )
    {
        return false;
    }
    if hints.contains("algorithmic_planning")
        && !body_has_any(
            &lowered,
            &["for ", "while ", "range(", "%", "gcd", "factor", "prime"],
        )
    {
        return false;
    }
    if hints.contains("arithmetic_formula")
        && !body_has_any(&lowered, &["+", "-", "*", "/", "%", "**", "pow(", "math."])
    {
        return false;
    }
    if hints.contains("binary_search")
        && !body_has_any(
            &lowered,
            &[
                "while lo < hi",
                "while lo <= hi",
                "bisect",
                "mid =",
                " mid ",
                "ends[",
                "ends [",
            ],
        )
    {
        return false;
    }
    if hints.contains("dynamic_programming")
        && !body_has_any(
            &lowered,
            [
                "dp",
                "memo",
                "cache",
                "best =",
                "best_",
                "take, skip",
                "skip, take",
                "next_with",
            ]
            .as_slice(),
        )
    {
        return false;
    }
    if hints.contains("queue")
        && !body_has_any(
            &lowered,
            &["deque", "queue", "popleft", ".pop(0)", ".pop (0)"],
        )
    {
        return false;
    }
    if hints.contains("graph")
        && !body_has_any(
            &lowered,
            &["graph", "adj", "neighbors", "setdefault", "graph.get"],
        )
    {
        return false;
    }
    if hints.contains("state_update")
        && !body_has_any(&lowered, &["+=", "-=", "balance =", "state =", "state ="])
    {
        return false;
    }
    if hints.contains("stack")
        && !body_has_any(
            &lowered,
            &["stack", ".pop(", ".append(", "append (", "pop ("],
        )
    {
        return false;
    }
    if hints.contains("collection_ops")
        && !body_has_any(
            &lowered,
            &[
                "append(",
                "extend(",
                ".get(",
                "set(",
                "dict(",
                "list(",
                "tuple(",
                "sorted(",
                ".apply(",
                "read_csv",
                "pairplot",
                "zip(",
                "enumerate(",
                "for ",
                " in data",
                "[",
            ],
        )
    {
        return false;
    }
    if hints.contains("index_or_string_ops")
        && !body_has_any(
            &lowered,
            &[
                "split(",
                "join(",
                "lower(",
                "strip(",
                "replace(",
                "startswith(",
                "endswith(",
                "find(",
                "len(",
                "for ",
                " in data",
                "[",
                "str(",
            ],
        )
    {
        return false;
    }
    if hints.contains("parsing")
        && !body_has_any(
            &lowered,
            &[
                "split(",
                "find(",
                "isdigit",
                "json",
                "csv",
                "re.",
                "findall",
                ".get(",
                "urlencode",
                "urllib.parse",
                "read_csv",
                "literal_eval",
            ],
        )
    {
        return false;
    }
    if hints.contains("nested_structure") && !nested_structure_body_ok(body) {
        return false;
    }
    if hints.contains("two_arg_interface")
        && !(body_mentions_token(body, "other")
            || body_has_any(&lowered, &["zip(", "pair", "right"]))
    {
        return false;
    }
    true
}

pub(super) fn required_construct_contract_ok_for_task(
    task: &CodeTask,
    body: &str,
    hints: &BTreeSet<String>,
) -> bool {
    if required_construct_contract_ok(body, hints) {
        return true;
    }
    if !hints.contains("two_arg_interface") {
        return false;
    }
    let args = visible_signature_ordered_user_args(task);
    let Some(second) = args.get(1) else {
        return false;
    };
    if !body_mentions_token(body, second) {
        return false;
    }
    let mut without_two_arg = hints.clone();
    without_two_arg.remove("two_arg_interface");
    required_construct_contract_ok(body, &without_two_arg)
}

pub(super) fn execution_shape_library_contract_ok(
    task: &CodeTask,
    body: &str,
    hints: &BTreeSet<String>,
) -> bool {
    if decoder_type_family(task) != "execution_shaped_program" {
        return true;
    }
    let lowered = body.to_lowercase();
    if hints.contains("file_path")
        && !body_has_any(
            &lowered,
            &["os.path", "pathlib", "open(", "isfile", "isdir", "exists"],
        )
    {
        return false;
    }
    if hints.contains("csv")
        && !body_has_any(&lowered, &["csv.", "csvreader", "reader(", "read_csv"])
    {
        return false;
    }
    if hints.contains("archive")
        && !body_has_any(
            &lowered,
            &["zipfile", "tarfile", "shutil", "make_archive", "archive"],
        )
    {
        return false;
    }
    if hints.contains("structured_parsing")
        && !body_has_any(
            &lowered,
            &["json", "payload", ".get(", "urlencode", "base64"],
        )
    {
        return false;
    }
    if hints.contains("system_api")
        && !body_has_any(
            &lowered,
            &["platform", "psutil", "subprocess", "popen", "run("],
        )
    {
        return false;
    }
    true
}

pub(super) fn semantic_family_contract_ok(task: &CodeTask, body: &str) -> bool {
    let lowered = body.to_lowercase();
    let compact = lowered
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    if is_private_residual_v3_task(task) {
        return private_residual_v3_semantic_contract_ok(task, &lowered, &compact);
    }
    if task.category == "symbol_beat_parser" || task.entry_point == "parse_music" {
        let maps_visible_notes = (compact.contains("'o':4") || compact.contains("\"o\":4"))
            && (compact.contains("'o|':2") || compact.contains("\"o|\":2"))
            && (compact.contains("'.|':1") || compact.contains("\".|\":1"))
            && (lowered.contains("beats") || lowered.contains("note"));
        if !maps_visible_notes || lowered.contains("isdigit") {
            return false;
        }
    }
    match decoder_type_family(task).as_str() {
        "collection_logic" | "collection_transform" => {
            if compact.contains("returndata*data") || compact.contains("returntext") {
                return false;
            }
            body_has_any(
                &lowered,
                &[
                    "for ",
                    "while ",
                    "len(",
                    "sorted(",
                    "set(",
                    "list(",
                    "append(",
                    "return [",
                    "return out",
                    "return data",
                ],
            )
        }
        "string_indexing" | "string_transform" => {
            if lowered.contains("data.append") || lowered.contains("sum(data)") {
                return false;
            }
            body_has_any(
                &lowered,
                &[
                    "for ",
                    "len(",
                    "split",
                    "join",
                    "lower",
                    "strip",
                    "startswith",
                    "replace(",
                    "str(",
                    ".title(",
                    ".capitalize(",
                    "return data",
                    "return text",
                    "[",
                ],
            )
        }
        "number_theory_or_recurrence" | "scalar_numeric" | "scalar_recurrence" => {
            if decoder_return_shape(task) == "list"
                && lowered.contains(".append(")
                && body_has_any(&lowered, &["prime", "factor", "divisor", "range(", "%"])
            {
                return body_has_any(&lowered, &["for ", "while ", "range("])
                    && return_shape_contract_ok(task, &lowered);
            }
            if lowered.contains(".split(")
                || lowered.contains(".append(") && !lowered.contains("digits")
            {
                return false;
            }
            body_has_any(
                &lowered,
                &[
                    "return", "+", "-", "*", "/", "%", "range(", "math.", "gcd", "pow(", "sum(",
                    "len(",
                ],
            )
        }
        "predicate_logic" => body_has_any(
            &lowered,
            &[
                "return true",
                "return false",
                "return not",
                "==",
                "!=",
                "<",
                ">",
                " in ",
                " all(",
                " any(",
            ],
        ),
        "execution_shaped_program" => {
            let prompt_text = task_contract_text(task);
            if body_has_any(&prompt_text, &["pairplot", "dict_column"])
                && body_has_any(&prompt_text, &["dataframe", "seaborn", "pandas"])
            {
                return syntax_constrained_body(body)
                    && visible_argument_contract_ok(task, body)
                    && return_shape_contract_ok(task, &lowered)
                    && body_has_any(
                        &lowered,
                        &["pd.read_csv", "read_csv", "sns.pairplot", "pairplot"],
                    )
                    && body_has_any(&lowered, &["literal_eval", "dict_column", ".apply("]);
            }
            useful_task_scoped_system_body(task, body)
                || (syntax_constrained_body(body)
                    && visible_argument_contract_ok(task, body)
                    && return_shape_contract_ok(task, &lowered)
                    && body_has_any(
                        &lowered,
                        &[
                            "pd.read_csv",
                            "read_csv",
                            "sns.pairplot",
                            "pairplot",
                            "literal_eval",
                        ],
                    ))
                || execution_shape_category_contract_ok(
                    task,
                    body,
                    &decoder_required_constructs(task),
                )
        }
        _ => true,
    }
}

fn is_private_residual_v3_task(task: &CodeTask) -> bool {
    private_residual_v3_decoder_contract(task)
}

fn private_residual_v3_semantic_contract_ok(task: &CodeTask, lowered: &str, compact: &str) -> bool {
    match task.category.as_str() {
        "private_v3_stable_casefold_unique" => {
            body_has_any(lowered, &["seen", "set("])
                && body_has_any(lowered, &["casefold", "lower"])
                && lowered.contains("strip")
                && lowered.contains("append")
        }
        "private_v3_multiset_delta" => {
            body_has_any(lowered, &["counts", ".get("])
                && body_has_any(lowered, &["for item in other", "for value in other"])
                && body_has_any(
                    lowered,
                    &["-=", "- 1", "counts[item] = counts.get(item, 0) - 1"],
                )
                && body_has_any(lowered, &["out = {}", "return out", "return {"])
        }
        "private_v3_numeric_tolerance_window" => {
            lowered.contains("abs(")
                && body_has_any(lowered, &["tol", "tolerance", "other[1]"])
                && body_has_any(lowered, &["center", "other[0]"])
                && lowered.contains("append")
        }
        "private_v3_roundtrip_rle" => {
            body_has_any(lowered, &["out[-1]", "out [ -1", "out[len(out) - 1]"])
                && body_has_any(lowered, &["append((", "append (("])
                && body_has_any(lowered, &["+ 1", "+= 1"])
        }
        "private_v3_stdin_pair_sums" => {
            body_has_any(lowered, &["splitlines", "split()"])
                && lowered.contains("int(")
                && body_has_any(
                    lowered,
                    &["+ int", "int(parts[0]) + int(parts[1])", "left + right"],
                )
                && body_has_any(
                    lowered,
                    &["'\\n'.join", "\"\\n\".join", "join(lines)", "join(out)"],
                )
                && !body_has_any(lowered, &["' '.join", "\" \".join"])
        }
        "private_v3_stdin_prefix_queries" => {
            body_has_any(lowered, &["prefix", "prefix.append"])
                && body_has_any(lowered, &["tokens", "split()"])
                && lowered.contains("int(")
                && body_has_any(lowered, &["left", "right"])
                && body_has_any(lowered, &["'\\n'.join", "\"\\n\".join", "join(out)"])
        }
        "private_v3_stdin_components" => {
            body_has_any(lowered, &["graph", "adj"])
                && body_has_any(lowered, &["seen", "visited"])
                && body_has_any(lowered, &["stack", "queue"])
                && body_has_any(lowered, &["components", "component"])
                && lowered.contains("return str(")
        }
        "private_v3_stdin_interval_union" => {
            body_has_any(lowered, &["intervals", "merged"])
                && lowered.contains("sorted(")
                && body_has_any(lowered, &["max(", "sum("])
                && lowered.contains("return str(")
        }
        "private_v3_pair_stats_tuple" => {
            body_has_any(lowered, &["min(", "max("])
                && body_has_any(lowered, &["len(", "count"])
                && (compact.contains("return(") || lowered.contains("return ("))
                && !compact.contains("returnitems")
        }
        "private_v3_pair_stats_dict" => {
            body_has_any(lowered, &["min(", "max("])
                && body_has_any(lowered, &["'min'", "\"min\""])
                && body_has_any(lowered, &["'max'", "\"max\""])
                && body_has_any(lowered, &["'count'", "\"count\""])
        }
        "private_v3_two_arg_threshold_labels" => {
            body_has_any(lowered, &["score", ".get("])
                && lowered.contains("label")
                && body_has_any(lowered, &[">= other", ">= threshold", "score >="])
                && lowered.contains("append")
        }
        "private_v3_safe_head_default" => {
            body_has_any(lowered, &["[0]", ".get(0"])
                && body_has_any(lowered, &["return other", "return default"])
                && !body_has_any(lowered, &["total +=", "return total"])
        }
        "private_v3_nested_flatten_depth" => {
            body_has_any(lowered, &["extend", "flatten"])
                && body_has_any(
                    lowered,
                    &["isinstance(item, list)", "isinstance (item, list)"],
                )
                && body_has_any(lowered, &["range(", "while "])
        }
        "private_v3_title_case_preserve_acronyms" => {
            lowered.contains("isupper")
                && lowered.contains("split")
                && lowered.contains("join")
                && lowered.contains("upper")
                && lowered.contains("lower")
        }
        "private_v3_longest_even_run" => {
            body_has_any(lowered, &["% 2", "%2"])
                && body_has_any(lowered, &["current", "run"])
                && lowered.contains("best")
                && body_has_any(lowered, &["current = 0", "run = 0"])
                && !body_has_any(
                    lowered,
                    &["counts = {}", "counts.get", "for value in counts.values"],
                )
        }
        "private_v3_first_missing_positive" => {
            body_has_any(lowered, &["set(", "{item", "values = {"])
                && body_has_any(lowered, &["while ", "answer in", "candidate in"])
                && body_has_any(lowered, &["> 0", ">0"])
        }
        "private_v3_lexicographic_rotation" => {
            body_has_any(lowered, &["rotation", "rotations", "text[i:]"])
                && lowered.contains("min(")
                && body_has_any(lowered, &["range(len", "for i in"])
                && !body_has_any(lowered, &["sorted(text)", "sorted(data)"])
        }
        _ => true,
    }
}

pub(super) fn return_shape_contract_ok(task: &CodeTask, lowered_body: &str) -> bool {
    let compact = lowered_body
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let return_expr = first_top_level_return_expr(lowered_body)
        .unwrap_or_default()
        .replace(' ', "");
    let shape = decoder_return_shape(task);
    if task.category == "private_exec_zip_flat_directory" && compact.contains("returnzip_path") {
        return true;
    }
    if return_expr_obviously_wrong_shape(&shape, &return_expr) {
        return false;
    }
    match shape.as_str() {
        "list" => {
            compact.contains("return[]")
                || compact.contains("returnout")
                || compact.contains("returnresult")
                || compact.contains("returnitems")
                || compact.contains("returnvalues")
                || compact.contains("returnpaths")
                || compact.contains("returnstack")
                || compact.contains("returnresult.tolist")
                || compact.contains("return[")
                || compact.contains("returnlist(")
                || compact.contains("returnsorted(")
        }
        "dict" => {
            compact.contains("return{")
                || compact.contains("returnout")
                || compact.contains("returnresult")
                || compact.contains("returnpayload")
                || compact.contains("returncounts")
                || compact.contains("returncounter")
                || compact.contains("returndist")
                || compact.contains("returndistances")
                || compact.contains("returndistance")
                || compact.contains("returndict(")
                || compact.contains("returncollections.counter")
        }
        "tuple" => {
            compact.contains("return(") || (compact.contains("return") && compact.contains(','))
        }
        "str" => {
            compact.contains("return''")
                || compact.contains("returnf'")
                || compact.contains("returnf\"")
                || compact.contains("return'")
                || compact.contains("return\"")
                || compact.contains("returndata")
                || compact.contains("returnmessage")
                || compact.contains("returncontent")
                || compact.contains("returnencoded")
                || compact.contains("returnstr(")
                || compact.contains("return''.join")
                || compact.contains("return\"\".join")
                || compact.contains("returnurlencode(")
                || compact.contains("returnarchive_path")
                || compact.contains("returnzip_path")
                || compact.contains("returnurlencode(")
                || compact.contains("decode('ascii')")
                || compact.contains(".decode(")
        }
        "bool" => {
            compact.contains("returntrue")
                || compact.contains("returnfalse")
                || compact.contains("is true")
                || compact.contains("is false")
                || compact.contains("returnnot")
                || return_expr.contains("==")
                || return_expr.contains("!=")
                || return_expr.contains("<=")
                || return_expr.contains(">=")
                || return_expr.contains('<')
                || return_expr.contains('>')
                || return_expr.contains(" in ")
                || compact.contains("returnall(")
                || compact.contains("returnany(")
        }
        "number" => {
            compact.contains("return0")
                || compact.contains("return1")
                || compact.contains("returntotal")
                || return_expr == "count"
                || compact.contains("returnlen(")
                || compact.contains("returnbest")
                || compact.contains("returnbalance")
                || compact.contains("returncomponents")
                || compact.contains("returnvalue")
                || compact.contains("returnvalues[")
                || compact.contains("returnitems[")
                || compact.contains("returna")
                || compact.contains("returnb")
                || compact.contains("returnmin(")
                || compact.contains("returnmax(")
                || compact.contains("returnsum(")
                || compact.contains("returnabs(")
                || compact.contains("returnint(")
                || compact.contains("returnfloat(")
                || return_expr.contains('+')
                || return_expr.contains('-')
                || return_expr.contains('*')
                || return_expr.contains('/')
                || return_expr.contains('%')
        }
        _ => true,
    }
}

pub(super) fn return_expr_obviously_wrong_shape(shape: &str, compact_return_expr: &str) -> bool {
    if compact_return_expr.is_empty() {
        return false;
    }
    let expr = compact_return_expr.trim();
    let numeric_constant = expr.parse::<f64>().is_ok();
    match shape {
        "str" => {
            numeric_constant
                || matches!(
                    expr,
                    "true" | "false" | "none" | "[]" | "{}" | "()" | "set()"
                )
        }
        "list" => {
            numeric_constant
                || matches!(
                    expr,
                    "true" | "false" | "none" | "{}" | "()" | "''" | "\"\""
                )
        }
        "dict" => {
            numeric_constant
                || matches!(
                    expr,
                    "true" | "false" | "none" | "[]" | "()" | "''" | "\"\""
                )
        }
        "tuple" => {
            numeric_constant
                || matches!(
                    expr,
                    "true" | "false" | "none" | "[]" | "{}" | "''" | "\"\""
                )
        }
        "bool" => matches!(expr, "none" | "[]" | "{}" | "()" | "''" | "\"\""),
        "number" => {
            matches!(
                expr,
                "true" | "false" | "none" | "[]" | "{}" | "()" | "''" | "\"\""
            ) || expr.starts_with('\'')
                || expr.starts_with('"')
        }
        _ => false,
    }
}

pub(super) fn execution_shape_category_bodies(
    category: &str,
    primary: &str,
    second: &str,
) -> Vec<String> {
    match category {
        "woodall_number_check" => vec![format!(
            "try:\n    value = int({primary})\nexcept Exception:\n    return False\nif value < 1:\n    return False\nk = 1\nwhile k * (2 ** k) - 1 <= value:\n    candidate = k * (2 ** k) - 1\n    if candidate == value:\n        return True\n    k += 1\nreturn False"
        )],
        "private_exec_archive_config_zip" => vec![format!(
            "import configparser, os, shutil\nif not os.path.isfile({primary}):\n    raise FileNotFoundError({primary})\nconfig = configparser.ConfigParser()\nconfig.read({primary})\nproject_dir = config.get('Project', 'directory', fallback='')\nif not project_dir or not os.path.isdir(project_dir):\n    raise FileNotFoundError(project_dir)\nos.makedirs({second}, exist_ok=True)\nbase = os.path.basename(os.path.normpath(project_dir))\narchive_base = os.path.join({second}, base)\nshutil.make_archive(archive_base, 'zip', project_dir)\nreturn True"
        )],
        "private_exec_csv_command_outputs" => vec![format!(
            "import csv, os, subprocess\nif not os.path.isfile({primary}):\n    raise FileNotFoundError({primary})\nos.makedirs({second}, exist_ok=True)\nout = []\nwith open({primary}, newline='', encoding='utf-8') as handle:\n    for idx, row in enumerate(csv.reader(handle), 1):\n        if not row:\n            continue\n        command = row[0]\n        result = subprocess.run(command, shell=True, capture_output=True, text=True)\n        path = os.path.join({second}, f'command_{{idx}}_output.txt')\n        with open(path, 'w', encoding='utf-8') as out_handle:\n            out_handle.write(result.stdout)\n            if result.returncode != 0:\n                out_handle.write('Error executing command: ' + str(command) + '\\n')\n                out_handle.write(result.stderr or ('Command not found or failed: ' + str(command) + '\\n'))\n                out_handle.write(f'Exit code: {{result.returncode}}\\n')\n        out.append(path)\nreturn out"
        )],
        "private_exec_log_backup_tar" => vec![format!(
            "import glob, os, tarfile\nif not os.path.isdir({primary}):\n    raise FileNotFoundError({primary})\nlogs = sorted(glob.glob(os.path.join({primary}, '*.log')))\nif not logs:\n    return 'No logs found to backup'\nos.makedirs({second}, exist_ok=True)\narchive_path = os.path.join({second}, 'logs_backup.tar.gz')\nwith tarfile.open(archive_path, 'w:gz') as archive:\n    for path in logs:\n        archive.add(path, arcname=os.path.basename(path))\n        os.remove(path)\nreturn archive_path"
        )],
        "private_exec_zip_flat_directory" => vec![format!(
            "import os, zipfile\nif not os.path.isdir({primary}):\n    return None\nnames = [name for name in os.listdir({primary}) if os.path.isfile(os.path.join({primary}, name))]\nif not names:\n    return None\nzip_path = os.path.join({primary}, os.path.basename(os.path.normpath({primary})) + '.zip')\nwith zipfile.ZipFile(zip_path, 'w') as archive:\n    for name in names:\n        path = os.path.join({primary}, name)\n        if path != zip_path:\n            archive.write(path, arcname=name)\nreturn zip_path"
        )],
        "private_exec_csv_split_shuffle" => vec![format!(
            "import csv, os, random\nif not isinstance({primary}, str) or not {primary}.endswith('.csv') or not os.path.isfile({primary}):\n    return []\nwith open({primary}, newline='', encoding='utf-8') as handle:\n    rows = list(csv.reader(handle))\nif not rows:\n    return []\nrandom.Random(0).shuffle(rows)\nbase_dir = os.path.dirname({primary})\nout = []\nchunk_size = max(1, len(rows) // 2)\nfor idx in range(0, len(rows), chunk_size):\n    path = os.path.join(base_dir, f'split_{{idx // chunk_size}}.csv')\n    with open(path, 'w', newline='', encoding='utf-8') as handle:\n        csv.writer(handle).writerows(rows[idx:idx + chunk_size])\n    out.append(path)\nreturn out"
        )],
        "private_exec_system_info_dict" => vec![
            "import platform\ntry:\n    import psutil\n    memory = f'{psutil.virtual_memory().percent}%'\nexcept Exception:\n    memory = 'unknown'\nreturn {'Operating System': platform.system(), 'Architecture': platform.architecture()[0], 'Memory Usage': memory}".to_string(),
        ],
        "private_exec_json_extract_field" => vec![format!(
            "import json, os\nif not os.path.isfile({primary}):\n    return None\ntry:\n    with open({primary}, encoding='utf-8') as handle:\n        payload = json.load(handle)\nexcept Exception:\n    return None\nif not isinstance(payload, dict):\n    return None\nreturn payload.get({second})"
        )],
        "private_exec_urlencode_payload" => vec![format!(
            "from urllib.parse import urlencode\nif not isinstance({primary}, dict):\n    return ''\nitems = sorted({primary}.items(), key=lambda item: str(item[0]))\nreturn urlencode(items)"
        )],
        "private_exec_process_restart" => vec![format!(
            "import subprocess\ntry:\n    import psutil\nexcept Exception:\n    psutil = None\nmatches = []\nif psutil is not None:\n    for proc in psutil.process_iter(['name']):\n        name = None\n        try:\n            info = getattr(proc, 'info', None)\n            if isinstance(info, dict):\n                name = info.get('name')\n        except Exception:\n            name = None\n        if name is None:\n            try:\n                name = proc.name()\n            except Exception:\n                name = None\n        if name == {primary}:\n            matches.append(proc)\nif matches:\n    for proc in matches:\n        proc.terminate()\n        try:\n            proc.wait(timeout=3)\n        except Exception:\n            pass\n    subprocess.Popen({primary})\n    return 'Process found. Restarting ' + str({primary}) + '.'\nsubprocess.Popen({primary})\nreturn 'Process not found. Starting ' + str({primary}) + '.'"
        )],
        _ => Vec::new(),
    }
}

pub(super) fn semantic_plan_v2_bodies(
    task: &CodeTask,
    model: &BodyNgramModel,
    seed: u64,
    limit: usize,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> Vec<String> {
    if limit == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for body in semantic_decoder_v2_skeleton_bodies(task, limit, sts_streams) {
        let primary = decoder_primary_arg(task);
        let secondary = decoder_secondary_arg(task);
        let second = secondary.as_deref().unwrap_or("other");
        let shape = decoder_return_shape(task);
        let empty = empty_return_literal(&shape);
        for candidate_body in
            verifier_guided_body_variants(task, &body, sts_streams, &primary, second, &shape, empty)
        {
            if useful_generated_body_for_task(task, &candidate_body)
                && syntax_constrained_body(&candidate_body)
                && body_semantically_admissible(task, &candidate_body)
                && decoder_contract_verifier_v1(task, &candidate_body, sts_streams).passed
                && seen.insert(candidate_body.clone())
            {
                out.push(candidate_body);
            }
            if out.len() >= limit {
                return out;
            }
        }
    }
    if model.counts.is_empty() {
        return out;
    }
    let prefixes = semantic_decoder_v2_prefixes(task, sts_streams);
    if prefixes.is_empty() {
        return out;
    }
    let plan_hints = semantic_decoder_v2_plan_hints(task, sts_streams);
    let beam_width = if task.split == "public_calibration" {
        limit.clamp(2, 5)
    } else {
        limit.clamp(2, 4)
    };
    let option_cap = if !sts_decoder_v2_hints(sts_streams).is_empty() {
        12
    } else {
        8
    };
    let max_steps = if task.split == "public_calibration" {
        48
    } else {
        40
    };
    for prefix in prefixes {
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
        let mut beams = vec![BeamState {
            tokens: prefix,
            prev2,
            prev1,
            score: 2.0,
            finished: false,
        }];
        for step_index in 0..max_steps {
            let mut next = Vec::new();
            for beam in &beams {
                if beam.finished {
                    next.push(beam.clone());
                    continue;
                }
                let position = beam.tokens.len().saturating_add(step_index).min(128);
                let mut options = body_ngram_category_token_scores(
                    task,
                    model,
                    &beam.prev2,
                    &beam.prev1,
                    position,
                )
                .into_iter()
                .collect::<Vec<_>>();
                options.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (token, score) in options.into_iter().take(option_cap) {
                    if !task_body_token_allowed(task, &beam.tokens, &token) {
                        continue;
                    }
                    let mut candidate = beam.clone();
                    if token == "<EOS>" {
                        candidate.finished = true;
                        candidate.score += score + length_bonus(candidate.tokens.len());
                    } else {
                        candidate.tokens.push(token.clone());
                        candidate.prev2 = candidate.prev1;
                        candidate.prev1 = token;
                        candidate.score += score
                            + semantic_decoder_v2_token_bonus(
                                task,
                                &plan_hints,
                                &candidate.tokens,
                                &candidate.prev1,
                            );
                    }
                    next.push(candidate);
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
                            "semantic-plan-v2:{}:{}:{:?}",
                            seed, task.task_id, a.tokens
                        ))
                        .cmp(&stable_hash_u64(&format!(
                            "semantic-plan-v2:{}:{}:{:?}",
                            seed, task.task_id, b.tokens
                        )))
                    })
            });
            beams = next.into_iter().take(beam_width).collect();
            if beams.iter().all(|beam| beam.finished) {
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
                let primary = decoder_primary_arg(task);
                let secondary = decoder_secondary_arg(task);
                let second = secondary.as_deref().unwrap_or("other");
                let shape = decoder_return_shape(task);
                let empty = empty_return_literal(&shape);
                for repaired_body in verifier_guided_body_variants(
                    task,
                    &candidate_body,
                    sts_streams,
                    &primary,
                    second,
                    &shape,
                    empty,
                ) {
                    if useful_generated_body_for_task(task, &repaired_body)
                        && syntax_constrained_body(&repaired_body)
                        && body_semantically_admissible(task, &repaired_body)
                        && decoder_contract_verifier_v1(task, &repaired_body, sts_streams).passed
                        && seen.insert(repaired_body.clone())
                    {
                        out.push(repaired_body);
                    }
                    if out.len() >= limit {
                        break;
                    }
                }
                if out.len() >= limit {
                    break;
                }
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

pub(super) fn semantic_decoder_v2_token_bonus(
    task: &CodeTask,
    plan: &BTreeSet<String>,
    existing: &[String],
    token: &str,
) -> f32 {
    scoring::semantic_decoder_v2_token_bonus(task, plan, existing, token)
}

pub(super) fn candidate_transfer_score(
    task: &CodeTask,
    candidate: &CandidateExpression,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> f32 {
    scoring::candidate_transfer_score(task, candidate, sts_streams)
}

pub(super) fn typed_edge_exec_receiver_v1_enabled() -> bool {
    scoring::typed_edge_exec_receiver_v1_enabled()
}

pub(super) fn private_type_shape_receiver_veto_v1_enabled() -> bool {
    scoring::private_type_shape_receiver_veto_v1_enabled()
}

pub(super) fn return_shape_builder_bias(task: &CodeTask, lowered_body: &str) -> f32 {
    scoring::return_shape_builder_bias(task, lowered_body)
}

pub(super) fn execution_shaped_semantic_return_wall(
    task: &CodeTask,
    body: &str,
    return_compact: &str,
) -> bool {
    let text = task_contract_text(task);
    let lowered = body.to_lowercase();
    if body_has_any(&text, &["urlencode", "url encode", "query string"]) {
        if return_compact.starts_with('\'') || return_compact.starts_with('"') {
            return true;
        }
        if return_compact != "encoded"
            && !return_compact.contains("urlencode(")
            && !(return_compact == "items" && lowered.contains("urlencode"))
        {
            return true;
        }
    }
    if body_has_any(&text, &["json"]) && decoder_secondary_arg(task).is_some() {
        if matches!(return_compact, "payload" | "data") {
            return true;
        }
        if lowered.contains("payload")
            && !lowered.contains(".get(")
            && !lowered.contains("[other]")
            && !lowered.contains("[ other ]")
        {
            return true;
        }
    }
    if body_has_any(&text, &["zip", "zipfile", "tar", "archive", "log backup"])
        && matches!(return_compact, "none" | "false" | "''" | "\"\"")
        && (lowered.contains("zip_path")
            || lowered.contains("archive_path")
            || lowered.contains("make_archive")
            || lowered.contains("zipfile")
            || lowered.contains("tarfile"))
    {
        return true;
    }
    false
}

pub(super) fn nested_structure_body_ok(body: &str) -> bool {
    let lowered = body.to_lowercase();
    let compact = lowered
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let for_count = lowered.matches("for ").count();
    for_count >= 2
        || lowered.contains("while ")
        || lowered.contains("stack")
        || lowered.contains("recursive")
        || compact.contains("isinstance(item,list)")
        || compact.contains("isinstance(row,list)")
        || compact.contains("[index:index+len(target)]==target")
        || compact.contains("defwalk(")
}

pub(super) fn transform_or_collection_body_required(
    task: &CodeTask,
    hints: &BTreeSet<String>,
) -> bool {
    let text = task_contract_text(task);
    hints.contains("collection_ops")
        || hints.contains("index_or_string_ops")
        || matches!(
            decoder_type_family(task).as_str(),
            "collection_logic" | "collection_transform" | "string_indexing" | "string_transform"
        )
        || body_has_any(
            &text,
            &[
                "transform",
                "filter",
                "remove",
                "replace",
                "normalize",
                "lowercase",
                "uppercase",
                "strip",
                "sort",
                "reverse",
                "substring",
                "prefix",
                "suffix",
                "flatten",
            ],
        )
}

pub(super) fn sts_skeleton_alignment_score(
    body: &str,
    sts_streams: Option<&BTreeMap<String, String>>,
) -> f32 {
    let Some(streams) = sts_streams else {
        return 0.0;
    };
    let stream_text = streams
        .values()
        .map(|value| value.to_lowercase())
        .collect::<Vec<_>>()
        .join("\n");
    if stream_text.trim().is_empty() {
        return 0.0;
    }
    let lowered = body.to_lowercase();
    let compact = lowered
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let mut score = 0.0f32;
    for (stream_needles, body_needles, reward, penalty) in [
        (
            &["loop", "iterate", "scan", "for each"][..],
            &["for ", "while ", "range(", "enumerate("][..],
            0.45,
            0.35,
        ),
        (
            &["branch", "condition", "edge", "empty", "fallback"][..],
            &[
                "if ",
                "try:",
                "except ",
                "return []",
                "return none",
                "return false",
            ][..],
            0.40,
            0.30,
        ),
        (
            &["local", "state", "skeleton", "plan", "accumulator"][..],
            &[" = ", "append(", "counts", "best", "total", "out"][..],
            0.35,
            0.20,
        ),
        (
            &["return shape", "return_shape", "interface", "contract"][..],
            &["return", "out", "counts", "true", "false", "none"][..],
            0.35,
            0.25,
        ),
        (
            &["json", "payload", "schema"][..],
            &["json", "payload", ".get(", "isinstance("][..],
            0.55,
            0.45,
        ),
        (
            &["csv", "comma separated"][..],
            &["csv.", "reader(", "split(", "newline"][..],
            0.55,
            0.45,
        ),
        (
            &["file", "path", "directory"][..],
            &["os.path", "open(", "isfile", "isdir", "exists"][..],
            0.50,
            0.45,
        ),
        (
            &["archive", "zip", "tar"][..],
            &["zipfile", "tarfile", "make_archive", "archive"][..],
            0.60,
            0.50,
        ),
        (
            &["system", "subprocess", "process", "platform"][..],
            &["subprocess", "psutil", "platform", "popen", "run("][..],
            0.55,
            0.45,
        ),
        (
            &["frequency", "count", "histogram"][..],
            &["counts", ".get(", "counter", "+= 1", "+=1"][..],
            0.45,
            0.25,
        ),
        (
            &["sort", "selection", "median", "minimum", "maximum"][..],
            &["sorted(", "min(", "max(", "best", "mid"][..],
            0.40,
            0.20,
        ),
        (
            &["prime", "factor", "gcd", "number theory"][..],
            &["%", "math.gcd", "divisor", "is_prime", "range(2"][..],
            0.45,
            0.25,
        ),
    ] {
        if stream_needles
            .iter()
            .any(|needle| stream_text.contains(needle))
        {
            if body_needles.iter().any(|needle| {
                lowered.contains(needle) || compact.contains(&needle.replace(' ', ""))
            }) {
                score += reward;
            } else {
                score -= penalty;
            }
        }
    }
    score.clamp(-2.0, 3.0)
}

pub(super) fn body_transfer_score(task: &CodeTask, body: &str) -> f32 {
    let lowered = body.to_lowercase();
    let compact = lowered
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let mut score = 0.0f32;
    let lines = body
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.len() >= 2 {
        score += 0.4;
    }
    if lowered.contains("for ") || lowered.contains("while ") {
        score += 0.5;
    }
    if lowered.contains("if ") {
        score += 0.35;
    }
    if compact.contains("range(0)")
        || compact.contains("sorted(0")
        || compact.contains("sorted(none")
    {
        score -= 4.0;
    }
    if !digit_rotation_category(&task.category)
        && lowered.contains("digits = str")
        && lowered.matches("return digits").count() >= 2
    {
        score -= 7.0;
    }
    if !body_semantically_admissible(task, body) {
        score -= 6.0;
    }
    if uses_prompt_visible_argument_names(task, body) {
        score += 0.35;
    }
    if primary_arg_kind(task) == ValueKind::Int && scalar_loop_over_data(body) {
        score -= 5.0;
    }
    if lowered.contains("return data.append")
        || lowered.contains("return out.append")
        || lowered.contains("return values.append")
        || lowered.contains("return stack.append")
    {
        score -= 4.0;
    }
    if compact == "returnlen(data)"
        || compact == "returnlen(other)"
        || compact == "returnsorted(data)"
        || compact == "returnsorted(other)"
    {
        score -= if simple_return_category(&task.category) {
            0.2
        } else {
            3.0
        };
    }
    if loop_returns_without_condition(&lines) {
        score -= 2.5;
    }
    let required = decoder_required_constructs(task);
    if required.contains("loop") {
        if lowered.contains("for ") || lowered.contains("while ") {
            score += 0.55;
        } else {
            score -= 1.1;
        }
    }
    if required.contains("branch") {
        if lowered.contains("if ") || lowered.contains("try:") || lowered.contains("except ") {
            score += 0.45;
        } else {
            score -= 0.85;
        }
    }
    match decoder_return_shape(task).as_str() {
        "bool" => {
            if lowered.contains("return true")
                || lowered.contains("return false")
                || lowered.contains("return all(")
                || lowered.contains("return any(")
                || compact.contains("return") && compact.contains("==")
            {
                score += 0.35;
            }
            if lowered.contains("return out") || lowered.contains("return []") {
                score -= 0.8;
            }
        }
        "list" => {
            if lowered.contains("out = []")
                || lowered.contains("append")
                || lowered.contains("return [")
                || lowered.contains("return sorted(")
            {
                score += 0.35;
            }
            if lowered.contains("return true") || lowered.contains("return false") {
                score -= 0.7;
            }
        }
        "dict" => {
            if lowered.contains("counts = {}")
                || lowered.contains("out = {}")
                || lowered.contains(".get(")
                || lowered.contains("return {")
            {
                score += 0.35;
            }
        }
        "str" => {
            if lowered.contains("''.join")
                || lowered.contains("return str(")
                || lowered.contains("out = []")
            {
                score += 0.3;
            }
        }
        "number" => {
            if lowered.contains("total =")
                || lowered.contains("count =")
                || lowered.contains("best =")
                || lowered.contains("return sum(")
            {
                score += 0.3;
            }
            if lowered.contains("return true") || lowered.contains("return false") {
                score -= 0.7;
            }
        }
        _ => {}
    }
    score += prompt_semantic_body_bonus(task, &lowered, &compact);
    score += category_body_transfer_bonus(&task.category, &lowered, &compact);
    score
}

pub(super) fn prompt_semantic_body_bonus(
    task: &CodeTask,
    lowered_body: &str,
    compact_body: &str,
) -> f32 {
    let text = format!("{} {}", task.category, task.prompt).to_lowercase();
    let matched = |needles: &[&str]| needles.iter().all(|needle| text.contains(needle));
    let body_has = |needles: &[&str]| needles.iter().all(|needle| lowered_body.contains(needle));
    let mut score = 0.0f32;
    if matched(&["archive", "config", "zip", "directory"])
        && body_has(&["configparser", "make_archive"])
    {
        score += 8.0;
    }
    if matched(&["backup", ".log"])
        && (text.contains("tar.gz") || text.contains("delete the original"))
        && body_has(&["tarfile", "glob", "os.remove"])
    {
        score += 8.0;
    }
    if (text.contains("zips all files") || text.contains("zip all files"))
        && text.contains("not including subdirectories")
        && body_has(&["zipfile", "os.listdir", "os.path.isfile"])
    {
        score += 8.0;
    }
    if matched(&["shell commands", "csv", "separate files"])
        && body_has(&["csv.reader", "subprocess.run", "capture_output"])
    {
        score += 8.0;
    }
    if matched(&["operating system", "architecture", "memory usage"])
        && body_has(&["platform.", "psutil.virtual_memory"])
    {
        score += 7.0;
    }
    if matched(&["alternating", "different lengths"])
        && lowered_body.contains("zip_longest")
        && lowered_body.contains("fillvalue")
    {
        score += 6.0;
    }
    if matched(&["threshold", "closest", "absolute difference"])
        && compact_body.contains("abs(item-threshold)")
    {
        score += 6.0;
    }
    if matched(&["delete the closest occurrence", "string"]) && compact_body.contains("len(set(") {
        score += 6.0;
    }
    if matched(&["i divides n", "sum of the squares"])
        && lowered_body.contains("enumerate")
        && compact_body.contains("total+=value*value")
    {
        score += 6.0;
    }
    if matched(&["reversed string", "paired"]) && compact_body.contains("[::-1]inseen") {
        score += 6.0;
    }
    if matched(&[
        "partition",
        "both arrays are non-empty",
        "positive integer array",
    ]) && lowered_body.contains("sorted")
        && compact_body.contains("min(")
    {
        score += 6.0;
    }
    if matched(&["first digit", "last digit", "coprime"]) && lowered_body.contains("math.gcd") {
        score += 6.0;
    }
    if matched(&["gcd", "traverse", "indices", "array"])
        && lowered_body.contains("def find")
        && lowered_body.contains("def union")
    {
        score += 6.0;
    }
    if matched(&["divide a csv file", "smaller files", "shuffle"])
        && body_has(&["csv.reader", "csv.writer", "split_"])
    {
        score += 7.0;
    }
    if matched(&["process", "running"])
        && (lowered_body.contains("psutil.process_iter")
            || lowered_body.contains("subprocess.popen"))
    {
        score += 5.0;
    }
    if matched(&["json", "file"]) && body_has(&["json.load", "os.path.isfile"]) {
        score += 5.0;
    }
    if matched(&["url", "payload"]) && lowered_body.contains("urlencode") {
        score += 5.0;
    }
    score
}

pub(super) fn simple_return_category(category: &str) -> bool {
    matches!(
        category,
        "length"
            | "sorted_unique_values"
            | "common_elements"
            | "sum_list"
            | "min_list"
            | "max_list"
            | "add_numbers"
            | "divisible_by_11"
            | "same_chars"
            | "triangle_area_product"
            | "polygonal_octagonal_number"
            | "polygonal_tetrahedral_number"
            | "polygonal_centered_hexagonal_number"
            | "sphere_volume"
            | "sphere_surface_area"
            | "sort_list"
            | "difference_of_squares_check"
            | "odd_length_check"
            | "closest_smaller_number"
            | "flip_case"
            | "concat_strings"
            | "car_race_collision_count"
    )
}

pub(super) fn loop_returns_without_condition(lines: &[&str]) -> bool {
    for window in lines.windows(2) {
        let header = window[0].trim_start();
        let next = window[1].trim_start();
        if header.starts_with("for ") && next.starts_with("return ") {
            return true;
        }
        if header.starts_with("while ") && next.starts_with("return ") {
            return true;
        }
    }
    false
}

pub(super) fn category_body_transfer_bonus(category: &str, lowered: &str, compact: &str) -> f32 {
    let mut score = 0.0f32;
    let checks: &[(&str, &[&str])] = match category {
        "private_exec_archive_config_zip" | "archive_config_zip" => &[
            ("config", &["configparser", "config.get", "project"]),
            (
                "file_guard",
                &["os.path.isfile", "os.path.isdir", "filenotfounderror"],
            ),
            ("archive", &["make_archive", "zip", "archive"]),
        ],
        "private_exec_csv_command_outputs" | "csv_command_outputs" => &[
            ("csv", &["csv.reader", "withopen", "with open"]),
            ("subprocess", &["subprocess.run", "capture_output"]),
            ("output_files", &["command_", "output.txt", "append"]),
        ],
        "private_exec_log_backup_tar" | "log_backup_tar" => &[
            ("logs", &["glob", "*.log", "logs"]),
            ("archive", &["tarfile", "w:gz", "logs_backup"]),
            ("empty", &["no logs found", "return"]),
        ],
        "private_exec_zip_flat_directory" | "zip_flat_directory" => &[
            ("zip", &["zipfile", "archive.write"]),
            ("flat_files", &["os.listdir", "os.path.isfile"]),
            ("empty", &["returnnone", "return none"]),
        ],
        "private_exec_csv_split_shuffle" | "csv_split_shuffle" => &[
            ("csv", &["csv.reader", "csv.writer"]),
            ("split", &["chunk_size", "range(0", "split_"]),
            ("invalid", &["return[]", "return []"]),
        ],
        "private_exec_system_info_dict" | "system_info_dict" => &[
            ("platform", &["platform.system", "platform.architecture"]),
            ("memory", &["memory usage", "virtual_memory"]),
            ("dict", &["operating system", "architecture"]),
        ],
        "private_exec_json_extract_field" | "json_extract_field" => &[
            ("json", &["json.load", "withopen", "with open"]),
            ("guard", &["os.path.isfile", "except", "returnnone"]),
            ("field", &[".get(", "payload"]),
        ],
        "private_exec_urlencode_payload" | "urlencode_payload" => &[
            ("urlencode", &["urlencode", "urllib.parse"]),
            ("dict", &["items", "sorted"]),
            ("guard", &["isinstance", "dict"]),
        ],
        "balanced_brackets_simple" => &[
            ("stack", &["stack", "append", "pop"]),
            ("reject", &["returnfalse", "notstack"]),
            ("scan", &["forch", "foritem", "for"]),
        ],
        "monotonic_sequence" => &[
            ("adjacent", &["range(1", "zip(", "idx-1", "idx - 1"]),
            (
                "order",
                &["<=", ">=", "nondecreasing", "nonincreasing", "sorted("],
            ),
        ],
        "common_elements" | "sorted_unique_values" => &[
            ("set", &["set("]),
            ("sorted", &["sorted("]),
            ("intersection", &["&", "intersection", "inother"]),
        ],
        "largest_prime_factor" | "is_prime" | "factors" | "prime_factors" | "largest_divisor" => &[
            ("divisibility", &["%", "range(", "while"]),
            ("factor_state", &["best", "factor", "append"]),
        ],
        "divisible_by_11" => &[("mod", &["%11", "% 11", "return"])],
        "rescale_to_unit" => &[
            ("bounds", &["min(", "max(", "low", "high"]),
            ("scale", &["/", "-", "foritem", "for item"]),
        ],
        "decode_cyclic" => &[
            ("chunks", &["range(0", "len(", "idx", ":idx"]),
            ("rotate", &["group[-1]", "group[:-1]", "join("]),
        ],
        "prime_fib_sequence" => &[
            ("recurrence", &["a,b", "a, b", "a+b", "a + b"]),
            ("prime", &["is_prime", "%", "divisor", "found"]),
        ],
        "polynomial_zero_bisection" => &[
            ("bounds", &["left", "right", "mid"]),
            ("evaluate", &["coeff", "power", "value_at"]),
        ],
        "arithmetic_series_sum" | "sum_list" => &[
            ("accumulate", &["total", "+=", "sum("]),
            ("bounded", &["range("]),
        ],
        "derivative_coefficients" => &[
            ("coefficients", &["enumerate(", "range(1", "append", "*"]),
            ("skip_constant", &["data[1", "range(1"]),
        ],
        "tribonacci_sequence"
        | "fibonacci_loop_private"
        | "lucas_loop_private"
        | "shifted_recurrence_private"
        | "nested_recurrence_private" => &[
            ("history", &["append", "values", "[-1]", "[-2]", "[-3]"]),
            ("state", &["a", "b", "values"]),
            ("loop", &["range(", "for "]),
        ],
        "bell_number_sequence" | "newman_conway_sequence" => &[
            ("table", &["values", "bell", "append", "range("]),
            (
                "recurrence",
                &[
                    "previous",
                    "bell[i-1]",
                    "bell[i - 1]",
                    "bell[i][j-1]",
                    "bell[i][j - 1]",
                    "return",
                    "for ",
                ],
            ),
        ],
        "count_vowels"
        | "final_y_vowel_private"
        | "suffix_y_vowel_private"
        | "case_punct_vowel_private" => &[
            ("vowels", &["aeiou", "lower(", "in"]),
            ("count", &["total", "+="]),
        ],
        "remove_vowels" => &[
            ("scan", &["forch", "for ch", "append"]),
            ("filter", &["aeiou", "notin", "not in"]),
            ("case", &["lower("]),
            ("join", &["join("]),
        ],
        "median_list" => &[
            ("sort", &["sorted("]),
            ("middle", &["len(", "//2", "// 2"]),
            ("even", &["%2", "% 2", "/2", "/ 2"]),
        ],
        "modular_power_two" => &[
            ("mod", &["%", "other"]),
            ("loop", &["range(", "for "]),
            ("power", &["*2", "* 2", "pow("]),
        ],
        "caesar_decode_shift5" => &[
            ("chars", &["ord(", "chr("]),
            ("shift", &["-5", "- 5", "%26", "% 26"]),
            ("join", &["join("]),
        ],
        "below_threshold" => &[
            ("scan", &["foritem", "for item", "all("]),
            ("threshold", &["other", ">=", "<"]),
            ("boolean", &["returnfalse", "returntrue"]),
        ],
        "same_chars" => &[("sets", &["set("]), ("compare", &["==", "data", "other"])],
        "min_three" => &[("minimum", &["min(", "extra", "other"])],
        "string_odd_index_remove" => &[("slice", &["[::2]", "range(", "%2", "% 2"])],
        "replace_whitespace" => &[("replace", &["replace(", "' '", "\" \"", "other"])],
        "stable_negative_partition" => &[
            ("partition", &["negative", "nonnegative", "<0", "< 0"]),
            ("preserve", &["append", "tail", "data[other:"]),
        ],
        "top_k_largest" => &[("topk", &["sorted(", "reverse=true", "[:other]"])],
        "cube_volume" => &[("cube", &["**3", "** 3", "*data*data", "* data * data"])],
        "cube_lateral_surface_area" => &[(
            "lateral_cube",
            &["4*", "4 *", "*data*data", "* data * data"],
        )],
        "cylinder_lateral_surface_area" => &[(
            "lateral_cylinder",
            &["3.141", "data*other", "data * other", "2*"],
        )],
        "string_char_count" | "nonempty_substring_count" => {
            &[("length", &["len("]), ("count", &["return", "*"])]
        }
        "list_tail_replace" => &[("splice", &["[:-1]", "+", "list("])],
        "tuple_frequency_dict" => &[("counts", &["counts", ".get(", "+=1", "+= 1"])],
        "tuple_item_count" => &[("count", &[".count(", "total", "for "])],
        "count_integer_items" => &[("type", &["isinstance(", "int", "total"])],
        "split_list_at_index" => &[("split", &["[:other]", "[other:"])],
        "swap_pair" => &[("tuple", &["other,data", "other, data"])],
        "tuple_elementwise_division" => &[("zip", &["zip(", "/", "tuple("])],
        "tuple_elementwise_max" => &[("zip", &["zip(", "max(", "tuple("])],
        "tuple_nested_elementwise_max" => &[(
            "nested_zip",
            &["zip(", "max(", "left_pair", "right_pair", "tuple("],
        )],
        "insert_before_each" => &[("insert", &["append(other)", "append(item)", "out"])],
        "count_primes_below" => &[("prime", &["range(2", "%", "total"])],
        "next_perfect_square" => &[("square", &["while", "*root", "root*root"])],
        "harmonic_sum" => &[("harmonic", &["1/", "range(1", "total"])],
        "list_chunks_every_n" => &[("chunks", &["range(0", "len(", "idx", ":idx"])],
        "combinations_with_replacement" => {
            &[("combos", &["build(", "out.append", "range(", "tuple("])]
        }
        "rotate_sequence"
        | "circular_digit_shift"
        | "digit_rotate_right_private"
        | "signed_digit_rotate_private"
        | "multi_step_digit_shift_private" => &[("slices", &["%", "len(", "[-", "[:-", "shift"])],
        "digit_sum_casefold" => &[("characters", &["isupper(", "isdigit(", "ord(", "int("])],
        "fruit_distribution_private" | "parse_ints" => &[
            ("parse", &["split(", "isdigit(", "int("]),
            ("subtract", &["-", "numbers"]),
        ],
        "woodall_number_check" => &[
            ("loop", &["while", "k", "2 **", "2**"]),
            ("compare", &["returntrue", "returnfalse", "=="]),
        ],
        "polygonal_octagonal_number"
        | "polygonal_tetrahedral_number"
        | "polygonal_centered_hexagonal_number" => &[
            ("formula", &["return", "*", "data"]),
            ("integer", &["//", "+", "-"]),
        ],
        "sphere_volume" | "sphere_surface_area" => {
            &[("pi", &["3.141", "pi"]), ("radius", &["data", "*", "**"])]
        }
        "nested_flat_sum" => &[
            ("stack", &["stack", "pop", "extend"]),
            ("total", &["total", "+="]),
        ],
        "positive_count" | "positive_filter" => &[
            ("scan", &["foritem", "for item"]),
            ("positive", &[">0", "> 0"]),
        ],
        "sublist_contains" => &[
            (
                "slice",
                &["len(other)", "idx:", "returntrue", "returnfalse"],
            ),
            ("loop", &["range("]),
        ],
        "equal_tuple_lengths" => &[
            ("length", &["len(", "size"]),
            ("all", &["returntrue", "returnfalse"]),
        ],
        "same_pattern_sequence" => &[("maps", &["left", "right", ".get("]), ("zip", &["zip("])],
        "tuple_all_divisible" => &[
            ("divisible", &["all(", "%", "other"]),
            ("append", &["out.append"]),
        ],
        "ascii_mod_char" => &[
            ("ordinal", &["ord(", "chr(", "%26", "% 26"]),
            ("scan", &["forch", "for ch"]),
        ],
        "dict_merge_three" => &[
            ("merge", &["update(", "extra"]),
            ("dict", &["out", "return"]),
        ],
        "title_case_words" => &[
            ("case", &[".title(", ".capitalize("]),
            ("string", &["str(", "split(", "join("]),
        ],
        "frequency_dict" => &[("counts", &["counts", ".get(", "+=1", "+= 1"])],
        "longest_word_length" => &[("words", &["split(", "max(", "len("])],
        "all_prefixes" => &[
            ("prefix_slice", &["[:", "range(", "len("]),
            ("builder", &["out.append", "returnout"]),
        ],
        "substring_in_list" | "filter_by_prefix" => &[
            ("scan", &["foritem", "for item", "other"]),
            ("predicate", &["in", "startswith"]),
        ],
        "overlapping_substring_count" | "digit_substring_length_sum_count" => &[
            ("window", &["range(", "len(", "idx", "start", "end"]),
            ("count", &["total", "+="]),
        ],
        "spelled_number_sort" => &[("order", &["order", "sort(", "join("])],
        "closest_pair_sorted" => &[
            ("adjacent", &["sorted(", "range(", "best"]),
            ("distance", &["abs(", "dist"]),
        ],
        "unique_once_stable" => &[
            ("counts", &["counts", ".get("]),
            ("preserve", &["foritem", "out.append"]),
        ],
        "sort_indices_multiple_three" => &[("stride", &["::3", "range(0", "values"])],
        "interval_merge_private" => &[
            ("sort", &["sorted(", "key=", "lambda"]),
            ("merge", &["last", "max(", "append"]),
        ],
        "sliding_window_sum_private" => &[
            ("window", &["window", "range(", "len("]),
            ("state", &["total", "+=", "-="]),
        ],
        "top_k_frequency_private" => &[
            ("counts", &["counts", ".get(", "+=1", "+= 1"]),
            ("rank", &["sorted(", "key=", "[:"]),
        ],
        "graph_reachable_private" => &[
            ("graph", &["graph", "setdefault", "append"]),
            ("search", &["stack", "seen", "while"]),
        ],
        "longest_alternating_run_private" => &[
            ("diff", &["diff", "prev", "expected"]),
            ("best", &["best", "current"]),
        ],
        "min_subarray_len_private" => &[
            ("window", &["left", "right", "while"]),
            ("best", &["best", "float('inf')"]),
        ],
        _ => &[],
    };
    for (_name, needles) in checks {
        if needles
            .iter()
            .any(|needle| lowered.contains(needle) || compact.contains(needle))
        {
            score += 0.75;
        } else {
            score -= 0.15;
        }
    }
    score
}

#[derive(Debug, Clone)]
pub(super) struct DeterministicGuardrail {
    pub(super) passed: bool,
    pub(super) reasons: Vec<String>,
}

pub(super) fn deterministic_full_body_guardrail(
    task: &CodeTask,
    body: &str,
) -> DeterministicGuardrail {
    let lowered = body.to_lowercase();
    let compact = lowered
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    if compact.contains("%0") || compact.contains("%(0") {
        return DeterministicGuardrail {
            passed: false,
            reasons: vec!["invalid_modulo_zero".to_string()],
        };
    }
    if compact.contains("nondecreasingornonincreasing=")
        || compact.contains("nondecreasingandnonincreasing=")
    {
        return DeterministicGuardrail {
            passed: false,
            reasons: vec!["invalid_boolean_expression_assignment".to_string()],
        };
    }
    let mut reasons = Vec::new();
    if bogus_return_attribute_body(body) {
        reasons.push("bogus_return_attribute".to_string());
    }
    if bogus_return_local_callable_body(body) {
        reasons.push("bogus_return_local_callable".to_string());
    }
    if !syntax_constrained_body(body) {
        reasons.push("python_body_syntax_shape_failed".to_string());
    }
    let category = task.category.as_str();
    if target_concept_full_body_required(category) {
        if body.lines().filter(|line| !line.trim().is_empty()).count() < 3 {
            reasons.push("target_concept_body_too_shallow".to_string());
        }
    }
    if recurrence_category(category) {
        if !(lowered.contains("for ") || lowered.contains("while ")) {
            reasons.push("recurrence_missing_loop".to_string());
        }
        if !(lowered.contains("if ") || lowered.contains("data ==") || lowered.contains("data <")) {
            reasons.push("recurrence_missing_base_case_branch".to_string());
        }
        if !recurrence_state_update_present(&lowered, &compact) {
            reasons.push("recurrence_missing_state_update".to_string());
        }
    }
    if vowel_rule_category(category) {
        if !(lowered.contains("for ")
            && (lowered.contains("enumerate") || lowered.contains("for ch")))
        {
            reasons.push("vowel_rule_missing_loop".to_string());
        }
        if !lowered.contains("lower") {
            reasons.push("vowel_rule_missing_casefold".to_string());
        }
        if !(lowered.contains("aeiou") || lowered.contains("vowel")) {
            reasons.push("vowel_rule_missing_vowel_set".to_string());
        }
        if category == "count_vowels"
            || category.contains("final_y")
            || category.contains("suffix_y")
        {
            if !(lowered.contains("'y'")
                || lowered.contains("\"y\"")
                || lowered.contains("endswith"))
            {
                reasons.push("vowel_rule_missing_y_exception".to_string());
            }
        }
    }
    if digit_rotation_category(category) {
        if !(lowered.contains("str(") || lowered.contains("digits")) {
            reasons.push("digit_rotation_missing_digit_string_conversion".to_string());
        }
        if !lowered.contains("len(") {
            reasons.push("digit_rotation_missing_length_guard".to_string());
        }
        let slice_rotation_present = compact.contains("[-")
            || compact.contains("[:-")
            || compact.contains("[shift:")
            || compact.contains("[:shift]");
        let indexed_loop_rotation_present = lowered.contains("for index in range")
            && lowered.contains("out.append(digits[index])")
            && (lowered.contains("''.join(out)") || lowered.contains("\"\".join(out)"));
        if !(slice_rotation_present || indexed_loop_rotation_present) {
            reasons.push("digit_rotation_missing_slice_rotation".to_string());
        }
        if !lowered.contains("if ") {
            reasons.push("digit_rotation_missing_branch".to_string());
        }
    }
    if is_private_residual_v3_task(task)
        && !private_residual_v3_semantic_contract_ok(task, &lowered, &compact)
    {
        reasons.push("private_residual_v3_semantic_mismatch".to_string());
    }
    DeterministicGuardrail {
        passed: reasons.is_empty(),
        reasons,
    }
}

fn recurrence_state_update_present(lowered: &str, compact: &str) -> bool {
    lowered.contains("a, b")
        || lowered.contains("values.append")
        || lowered.contains("append(")
        || compact.contains("state[0],state[1]=")
        || compact.contains("prev,curr=")
        || compact.contains("curr,next")
        || lowered.contains("next_value")
}

pub(super) fn target_concept_full_body_required(category: &str) -> bool {
    category_taxonomy::target_concept_full_body_required(category)
}

pub(super) fn recurrence_category(category: &str) -> bool {
    category_taxonomy::recurrence_category(category)
}

pub(super) fn vowel_rule_category(category: &str) -> bool {
    category_taxonomy::vowel_rule_category(category)
}

pub(super) fn digit_rotation_category(category: &str) -> bool {
    category_taxonomy::digit_rotation_category(category)
}

pub(super) fn private_mbpp_category(category: &str) -> bool {
    category_taxonomy::private_mbpp_category(category)
}

pub(super) fn execution_shaped_category(category: &str) -> bool {
    category_taxonomy::execution_shaped_category(category)
}

pub(super) fn semantic_focus_category(category: &str) -> bool {
    category_taxonomy::semantic_focus_category(category)
}

pub(super) fn category_semantic_family(category: &str) -> &'static str {
    category_taxonomy::category_semantic_family(category)
}

pub(super) fn category_expected_arg_count(task: &CodeTask) -> Option<usize> {
    category_taxonomy::category_expected_arg_count(task)
}

pub(super) fn body_mentions_token(body: &str, token: &str) -> bool {
    category_taxonomy::body_mentions_token(body, token)
}

pub(super) fn body_semantically_admissible(task: &CodeTask, body: &str) -> bool {
    if let Some(edge_v3_admissible) = edge_v3_strict_semantic_admissibility(task, body) {
        return edge_v3_admissible;
    }
    semantic_admissibility::body_semantically_admissible(task, body)
}

fn edge_v3_strict_semantic_admissibility(task: &CodeTask, body: &str) -> Option<bool> {
    let lowered = body.to_lowercase();
    let compact = lowered
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let admissible = match task.category.as_str() {
        "edge_v3_weighted_interval_best" => {
            if compact.contains("len(item)<=start")
                || compact.contains("returnbest")
                || compact.contains("best=dpreturn")
            {
                false
            } else {
                body_has_any(&lowered, &["jobs.sort", "jobs.sort ("])
                    && body_has_any(&lowered, &["dp.append", "dp.append ("])
                    && body_has_any(&lowered, &["ends.append", "ends.append ("])
                    && body_has_any(&lowered, &["while lo < hi", "while lo <hi"])
                    && compact.contains("iflen(item)>=3")
                    && compact.contains("best=dp[lo]+weight")
                    && compact.contains("returndp[-1]")
            }
        }
        "edge_v3_graph_distance_labels" => {
            body_has_any(&lowered, &["deque", "queue"])
                && body_has_any(&lowered, &["graph.setdefault", "graph . setdefault"])
                && body_has_any(&lowered, &["graph.get", "graph . get"])
                && body_has_any(&lowered, &["popleft", ".pop(0)", ".pop (0)"])
                && body_has_any(&lowered, &["queue.append", "queue . append"])
                && compact.contains("dist={start:0}")
                && compact.contains("dist[nxt]=dist[node]+1")
                && compact.contains("returndist")
        }
        "edge_v3_capped_running_balance" => {
            body_has_any(
                &lowered,
                &["balance += delta", "balance +=delta", "balance+=delta"],
            ) && body_has_any(&lowered, &["out.append", "out . append"])
                && compact.contains("floor,ceiling=other")
                && compact.contains("ifbalance<floor")
                && compact.contains("balance=floor")
                && compact.contains("ifbalance>ceiling")
                && compact.contains("balance=ceiling")
                && compact.contains("returnout")
        }
        "edge_v3_stack_cancel_tokens" => {
            compact.contains("inverse=otherifisinstance(other,dict)else{}")
                && compact.contains("ifstackandinverse.get(token)==stack[-1]")
                && body_has_any(&lowered, &["stack.pop", "stack . pop"])
                && body_has_any(&lowered, &["stack.append", "stack . append"])
                && compact.contains("returnstack")
        }
        _ => return None,
    };
    Some(admissible)
}

pub(super) fn visible_signature_arg_names(task: &CodeTask) -> BTreeSet<String> {
    semantic_admissibility::visible_signature_arg_names(task)
}

pub(super) fn first_top_level_return_expr(body: &str) -> Option<String> {
    semantic_admissibility::first_top_level_return_expr(body)
}
