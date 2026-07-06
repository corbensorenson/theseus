// State-sequence decoder feature construction.
// Static task/prompt/contract features are built once per task; dynamic beam
// features stay cheap and local to each decode/training step.

use super::*;

pub(super) fn state_sequence_static_features(
    task: &CodeTask,
    prompt_tokens: &BTreeSet<String>,
) -> Vec<(String, f32)> {
    let mut features = vec![
        ("bias".to_string(), 1.0),
        (format!("category:{}", task.category), 2.6),
        (
            format!(
                "semantic_family:{}",
                category_semantic_family(&task.category)
            ),
            1.4,
        ),
    ];
    if let Some(arg_count) = category_expected_arg_count(task) {
        features.push((format!("expected_arg_count:{arg_count}"), 1.0));
    }
    if semantic_focus_category(&task.category) {
        features.push(("semantic_focus_category".to_string(), 1.0));
    }
    features.extend(decoder_contract_features(task));
    for token in prompt_tokens.iter().take(48) {
        features.push((format!("prompt:{token}"), 0.12));
        features.push((format!("category_prompt:{}|{token}", task.category), 0.1));
    }
    features
}

pub(super) fn state_sequence_features_with_static(
    task: &CodeTask,
    static_features: &[(String, f32)],
    existing: &[String],
    position: usize,
) -> Vec<(String, f32)> {
    let dynamic_features = state_sequence_dynamic_features(task, existing, position);
    let mut features = Vec::with_capacity(static_features.len() + dynamic_features.len());
    features.extend_from_slice(static_features);
    features.extend(dynamic_features);
    features
}

pub(super) fn state_sequence_dynamic_features(
    task: &CodeTask,
    existing: &[String],
    position: usize,
) -> Vec<(String, f32)> {
    let prev1 = existing
        .last()
        .map(String::as_str)
        .unwrap_or("<BOS>")
        .to_string();
    let prev2 = existing
        .iter()
        .rev()
        .nth(1)
        .map(String::as_str)
        .unwrap_or("<BOS>")
        .to_string();
    let prev3 = existing
        .iter()
        .rev()
        .nth(2)
        .map(String::as_str)
        .unwrap_or("<BOS>")
        .to_string();
    let indent = body_indent_balance(existing).min(4);
    let line_tokens = current_line_tokens(existing);
    let line_start =
        existing.is_empty() || matches!(prev1.as_str(), "<NL>" | "<INDENT>" | "<DEDENT>");
    let capped_position = learned_position_cap(&task.category, position);
    let mut features = Vec::with_capacity(48);
    features.extend([
        (format!("pos:{capped_position}"), 1.15),
        (format!("pos_bucket:{}", position / 4), 0.7),
        (format!("indent:{indent}"), 0.7),
        (format!("prev1:{prev1}"), 1.7),
        (format!("prev2:{prev2}"), 0.9),
        (format!("prev3:{prev3}"), 0.45),
        (format!("prev2_1:{prev2}|{prev1}"), 1.2),
        (format!("prev3_2_1:{prev3}|{prev2}|{prev1}"), 0.7),
        (format!("category_prev1:{}|{prev1}", task.category), 2.0),
        (
            format!("category_prev2_1:{}|{prev2}|{prev1}", task.category),
            1.6,
        ),
        (
            format!("category_pos:{}|{}", task.category, capped_position),
            2.2,
        ),
    ]);
    if existing.is_empty() {
        features.push(("slot:body_start".to_string(), 1.2));
        features.push((format!("category_body_start:{}", task.category), 1.3));
    }
    if line_start {
        features.push(("slot:line_start".to_string(), 1.0));
        features.push((format!("category_line_start:{}", task.category), 1.5));
    } else {
        features.push(("slot:line_continue".to_string(), 0.6));
    }
    if prev1 == ":" {
        features.push(("after_colon".to_string(), 1.4));
    }
    if previous_meaningful_token(existing) == Some(":".to_string()) && prev1 == "<NL>" {
        features.push(("line_after_block_header".to_string(), 1.4));
    }
    for token in line_tokens.iter().take(12) {
        features.push((format!("line_has:{token}"), 0.35));
        features.push((format!("category_line_has:{}|{token}", task.category), 0.35));
    }
    features
}

fn decoder_contract_features(task: &CodeTask) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    let shape = decoder_return_shape(task);
    if shape != "unknown" {
        out.push((format!("contract:return_shape:{shape}"), 1.5));
        out.push((
            format!("category_return_shape:{}|{}", task.category, shape),
            1.3,
        ));
    }
    let family = decoder_type_family(task);
    if family != "unknown" {
        out.push((format!("contract:type_family:{family}"), 1.15));
        out.push((
            format!("category_type_family:{}|{family}", task.category),
            1.0,
        ));
    }
    if let Some(arg_count) = category_expected_arg_count(task) {
        out.push((format!("contract:arg_count:{arg_count}"), 1.05));
    }
    for arg_name in visible_signature_arg_names(task).into_iter().take(6) {
        out.push((format!("contract:arg_name:{arg_name}"), 0.8));
        out.push((
            format!("category_arg_name:{}|{arg_name}", task.category),
            0.55,
        ));
    }
    for (idx, kind) in visible_arg_kinds(&task.entry_point, &task.prompt)
        .into_iter()
        .enumerate()
        .take(6)
    {
        let kind = format!("{:?}", kind).to_lowercase();
        out.push((format!("contract:arg_kind:{idx}:{kind}"), 0.85));
        out.push((
            format!("category_arg_kind:{}|{idx}:{kind}", task.category),
            0.6,
        ));
    }
    for item in decoder_required_constructs(task).into_iter().take(12) {
        out.push((format!("contract:requires:{item}"), 1.2));
        out.push((format!("category_requires:{}|{item}", task.category), 1.05));
    }
    for hint in semantic_decoder_v2_plan_hints(task, None)
        .into_iter()
        .take(32)
    {
        out.push((format!("contract:generation_hint:{hint}"), 1.05));
        out.push((
            format!("category_generation_hint:{}|{hint}", task.category),
            0.85,
        ));
    }
    if let Some(contract) = task.raw.get("decoder_contract").and_then(Value::as_object) {
        if let Some(label) = contract.get("residual_label_hint").and_then(Value::as_str) {
            out.push((format!("contract:residual:{label}"), 1.0));
        }
        if let Some(count) = contract
            .get("visible_arg_count_hint")
            .and_then(Value::as_i64)
        {
            out.push((format!("contract:visible_arg_count_hint:{count}"), 0.9));
        }
    }
    out
}

pub(super) fn decoder_return_shape(task: &CodeTask) -> String {
    let explicit = task
        .raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("return_shape"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    if broad_private_generated_decoder_contract(task) || private_residual_v3_decoder_contract(task)
    {
        return explicit;
    }
    let visible_signature_shape = visible_signature_return_shape_hint(task);
    if visible_signature_shape != "unknown" {
        return visible_signature_shape;
    }
    if explicit != "unknown" {
        explicit
    } else if category_return_shape_is_mixed(task) {
        "unknown".to_string()
    } else {
        let category_override = category_return_shape_override(task);
        if category_override != "unknown" {
            category_override.to_string()
        } else {
            visible_return_shape_hint(task)
        }
    }
}

fn category_return_shape_override(task: &CodeTask) -> &'static str {
    let text = task_contract_text(task);
    if matches!(
        task.category.as_str(),
        "below_threshold"
            | "same_chars"
            | "opposite_signs"
            | "palindrome"
            | "is_prime"
            | "is_anagram"
            | "dict_required_keys"
            | "balanced_brackets_simple"
            | "monotonic_sequence"
            | "two_sum_zero_exists"
            | "three_sum_zero_exists"
            | "divisible_by_11"
            | "palindrome_list_weight"
            | "multiply_three_primes"
            | "sublist_contains"
            | "private_mbpp_sublist_contains"
            | "equal_tuple_lengths"
            | "difference_of_squares_check"
            | "same_pattern_sequence"
            | "odd_length_check"
            | "private_exec_archive_config_zip"
    ) || text.contains("return whether")
        || text.contains("whether ")
        || text.contains("true if")
        || text.contains("false otherwise")
        || text.contains("boolean")
        || text.contains("returns:\n    - bool")
        || text.contains("returns:\n- bool")
        || text.contains("bool")
        || text.contains("check if")
    {
        return "bool";
    }
    if matches!(
        task.category.as_str(),
        "sum_list"
            | "add_numbers"
            | "abs_diff"
            | "max_list"
            | "min_list"
            | "min_three"
            | "median_list"
            | "median_odd"
            | "length"
            | "distinct_count"
            | "string_char_count"
            | "nonempty_substring_count"
            | "tuple_item_count"
            | "count_integer_items"
            | "count_primes_below"
            | "count_truthy"
            | "count_vowels"
            | "final_y_vowel_private"
            | "suffix_y_vowel_private"
            | "case_punct_vowel_private"
            | "word_count"
            | "positive_count"
            | "negative_count"
            | "count_digit_under_divisibility"
            | "private_mbpp_bell_number_small"
            | "bell_number_sequence"
            | "private_mbpp_hex_digit_count"
            | "hex_prime_count"
            | "arithmetic_series_sum"
            | "harmonic_sum"
            | "frequency_at_least_value"
            | "index_or_minus_one"
            | "largest_divisor"
            | "largest_prime_factor"
            | "next_perfect_square"
            | "gcd_pair"
            | "modular_power_two"
            | "triangle_area_product"
            | "triangle_area_sides"
            | "cube_volume"
            | "cube_lateral_surface_area"
            | "cylinder_lateral_surface_area"
            | "sphere_volume"
            | "sphere_surface_area"
            | "polygonal_octagonal_number"
            | "polygonal_tetrahedral_number"
            | "polygonal_centered_hexagonal_number"
            | "tribonacci_sequence"
            | "fibonacci_loop_private"
            | "lucas_loop_private"
            | "shifted_recurrence_private"
            | "nested_recurrence_private"
            | "polynomial_zero_bisection"
    ) {
        return "number";
    }
    if matches!(
        task.category.as_str(),
        "common_elements"
            | "sorted_unique_values"
            | "all_prefixes"
            | "filter_integers"
            | "list_tail_replace"
            | "stable_negative_partition"
            | "top_k_largest"
            | "flatten_once"
            | "insert_before_each"
            | "list_chunks_every_n"
            | "prime_factors"
            | "factors"
            | "derivative_coefficients"
            | "positive_filter"
            | "sort_by_second"
            | "private_mbpp_sort_pairs_by_second"
            | "stable_dedupe"
            | "matrix_diagonal"
            | "transpose_matrix"
            | "remove_none"
            | "parse_ints"
            | "symbol_beat_parser"
            | "powers_of_two"
            | "pluck_smallest_even"
            | "alternating_min_max_sort"
            | "total_match_lengths"
            | "tuple_all_divisible"
            | "unique_once_stable"
            | "filter_by_prefix"
            | "sort_indices_multiple_three"
            | "private_residual_list_tail_replace"
            | "private_exec_csv_command_outputs"
            | "private_exec_csv_split_shuffle"
    ) {
        return "list";
    }
    if matches!(
        task.category.as_str(),
        "caesar_decode_shift5"
            | "remove_vowels"
            | "reverse_string"
            | "string_sequence"
            | "base_digits"
            | "circular_digit_shift"
            | "decode_cyclic"
            | "parse_ints_text"
            | "extract_def_name"
            | "private_extract_first_def"
            | "replace_whitespace"
            | "spelled_number_sort"
            | "flip_case"
            | "concat_strings"
            | "ascii_mod_char"
            | "private_intended_lexicographic_run_decrement"
            | "digit_rotate_right_private"
            | "signed_digit_rotate_private"
            | "multi_step_digit_shift_private"
            | "private_digit_rotate_right"
            | "private_signed_digit_rotate"
            | "private_multi_step_digit_shift"
            | "private_overshift_reverse_digits"
            | "normalize_string"
            | "remove_spaces"
            | "title_case_words"
            | "private_reverse_text"
            | "private_exec_log_backup_tar"
            | "private_exec_urlencode_payload"
            | "private_exec_zip_flat_directory"
    ) {
        return "str";
    }
    if matches!(
        task.category.as_str(),
        "tuple_frequency_dict"
            | "frequency_dict"
            | "dict_merge_three"
            | "private_exec_system_info_dict"
    ) {
        return "dict";
    }
    if matches!(
        task.category.as_str(),
        "split_list_at_index"
            | "swap_pair"
            | "closest_pair"
            | "closest_pair_sorted"
            | "tuple_elementwise_division"
            | "tuple_elementwise_max"
            | "tuple_nested_elementwise_max"
    ) {
        return "tuple";
    }
    "unknown"
}

fn category_return_shape_is_mixed(task: &CodeTask) -> bool {
    matches!(
        task.category.as_str(),
        "rotate_sequence" | "take_every_other" | "safe_head"
    )
}
