use super::*;

pub(super) fn target_concept_full_body_required(category: &str) -> bool {
    recurrence_category(category)
        || vowel_rule_category(category)
        || digit_rotation_category(category)
        || private_mbpp_category(category)
        || execution_shaped_category(category)
}

pub(super) fn recurrence_category(category: &str) -> bool {
    matches!(
        category,
        "tribonacci_sequence"
            | "fibonacci_loop_private"
            | "lucas_loop_private"
            | "shifted_recurrence_private"
            | "nested_recurrence_private"
            | "private_fibonacci_loop"
            | "private_lucas_loop"
            | "private_shifted_recurrence"
            | "private_nested_recurrence"
    ) || category.contains("recurrence")
}

pub(super) fn vowel_rule_category(category: &str) -> bool {
    matches!(
        category,
        "count_vowels"
            | "final_y_vowel_private"
            | "suffix_y_vowel_private"
            | "case_punct_vowel_private"
            | "private_final_y_vowels"
            | "private_suffix_y_vowels"
            | "private_case_punct_vowels"
    ) || (category.contains("vowel") && !category.contains("remove"))
}

pub(super) fn digit_rotation_category(category: &str) -> bool {
    matches!(
        category,
        "circular_digit_shift"
            | "digit_rotate_right_private"
            | "signed_digit_rotate_private"
            | "multi_step_digit_shift_private"
            | "private_digit_rotate_right"
            | "private_signed_digit_rotate"
            | "private_overshift_reverse_digits"
            | "private_multi_step_digit_shift"
    ) || category.contains("digit_rotate")
        || category.contains("digit_shift")
}

pub(super) fn private_mbpp_category(category: &str) -> bool {
    category.starts_with("private_mbpp_")
}

pub(super) fn execution_shaped_category(category: &str) -> bool {
    category.starts_with("private_exec_")
        || matches!(
            category,
            "archive_config_zip"
                | "csv_command_outputs"
                | "csv_file_read"
                | "dataframe_transform"
                | "file_path_transform"
                | "json_file_transform"
                | "log_backup_tar"
                | "csv_split_shuffle"
                | "zip_flat_directory"
                | "archive_file_transform"
                | "system_info_dict"
                | "json_extract_field"
                | "urlencode_payload"
                | "process_restart"
        )
}

pub(super) fn private_mbpp_collection_category(category: &str) -> bool {
    matches!(
        category,
        "private_mbpp_filter_below_threshold"
            | "private_mbpp_all_below_threshold"
            | "private_mbpp_frequency_at_least"
            | "private_mbpp_select_pair_values"
            | "private_mbpp_common_sorted_unique"
            | "private_mbpp_filtered_median"
            | "private_mbpp_sort_pairs_by_second"
            | "private_mbpp_sublist_contains"
            | "private_mbpp_equal_tuple_lengths"
            | "private_mbpp_same_pattern_sequence"
            | "private_mbpp_monotonic_sequence"
    )
}

pub(super) fn private_mbpp_numeric_category(category: &str) -> bool {
    matches!(
        category,
        "private_mbpp_triangle_kind"
            | "private_mbpp_largest_prime_factor"
            | "private_mbpp_bell_number_small"
            | "private_mbpp_newman_conway_small"
    )
}

pub(super) fn private_mbpp_string_category(category: &str) -> bool {
    matches!(
        category,
        "private_mbpp_hex_digit_count"
            | "private_mbpp_parse_unit_total"
            | "private_mbpp_largest_concat"
    )
}

pub(super) fn semantic_focus_category(category: &str) -> bool {
    recurrence_category(category)
        || vowel_rule_category(category)
        || digit_rotation_category(category)
        || private_mbpp_category(category)
        || execution_shaped_category(category)
        || matches!(
            category,
            "median_list"
                | "below_threshold"
                | "same_chars"
                | "common_elements"
                | "index_or_minus_one"
                | "extract_def_name"
                | "reverse_string"
                | "palindrome"
                | "max_list"
                | "min_list"
                | "sum_list"
                | "opposite_signs"
                | "count_digit_under_divisibility"
                | "divisible_by_11"
                | "two_sum_zero_exists"
                | "three_sum_zero_exists"
                | "filter_integers"
                | "safe_head"
                | "dict_required_keys"
                | "parse_ints"
                | "min_three"
                | "string_odd_index_remove"
                | "replace_whitespace"
                | "stable_negative_partition"
                | "top_k_largest"
                | "cube_volume"
                | "cube_lateral_surface_area"
                | "cylinder_lateral_surface_area"
                | "string_char_count"
                | "nonempty_substring_count"
                | "list_tail_replace"
                | "tuple_frequency_dict"
                | "tuple_item_count"
                | "count_integer_items"
                | "split_list_at_index"
                | "swap_pair"
                | "tuple_elementwise_division"
                | "tuple_elementwise_max"
                | "tuple_nested_elementwise_max"
                | "insert_before_each"
                | "count_primes_below"
                | "next_perfect_square"
                | "harmonic_sum"
                | "list_chunks_every_n"
                | "combinations_with_replacement"
                | "triangle_area_sides"
                | "uppercase_ascii_sum"
                | "pluck_smallest_even"
                | "frequency_at_least_value"
                | "alternating_min_max_sort"
                | "palindrome_list_weight"
                | "smallest_palindrome_changes"
                | "total_match_lengths"
                | "multiply_three_primes"
                | "simple_power"
                | "cube_number"
                | "hex_prime_count"
                | "sort_by_second"
                | "rescale_to_unit"
                | "decode_cyclic"
                | "prime_fib_sequence"
                | "polynomial_zero_bisection"
                | "prime_factors"
                | "max_tuple_difference"
                | "woodall_number_check"
                | "polygonal_octagonal_number"
                | "polygonal_tetrahedral_number"
                | "polygonal_centered_hexagonal_number"
                | "sphere_volume"
                | "sphere_surface_area"
                | "nested_flat_sum"
                | "positive_count"
                | "positive_filter"
                | "sublist_contains"
                | "equal_tuple_lengths"
                | "sort_list"
                | "difference_of_squares_check"
                | "same_pattern_sequence"
                | "tuple_all_divisible"
                | "odd_length_check"
                | "ascii_mod_char"
                | "dict_merge_three"
                | "frequency_dict"
                | "closest_smaller_number"
                | "longest_word_length"
                | "substring_in_list"
                | "overlapping_substring_count"
                | "spelled_number_sort"
                | "closest_pair_sorted"
                | "unique_once_stable"
                | "flip_case"
                | "concat_strings"
                | "filter_by_prefix"
                | "sort_indices_multiple_three"
                | "car_race_collision_count"
                | "digit_substring_length_sum_count"
                | "bell_number_sequence"
                | "newman_conway_sequence"
                | "numeric_string_parser"
                | "private_index_or_default"
                | "private_extract_first_def"
                | "private_reverse_text"
                | "private_max_item"
                | "private_median_even_odd"
                | "private_all_below_threshold"
                | "private_opposite_signs"
                | "private_palindrome_check"
                | "private_decode_shift_general"
                | "private_pair_sum"
                | "private_same_char_set"
                | "private_gcd_pair_loop"
                | "private_prime_loop"
                | "private_anagram_sorted"
                | "private_base_digits"
                | "private_base_digits_alt"
                | "sliding_window_sum_private"
                | "longest_alternating_run_private"
                | "min_subarray_len_private"
        )
}

pub(super) fn category_semantic_family(category: &str) -> &'static str {
    if execution_shaped_category(category) {
        "execution_shaped_program"
    } else if recurrence_category(category) {
        "scalar_recurrence"
    } else if vowel_rule_category(category)
        || private_mbpp_string_category(category)
        || matches!(
            category,
            "reverse_string"
                | "palindrome"
                | "normalize_string"
                | "word_count"
                | "extract_def_name"
                | "parse_ints"
                | "string_odd_index_remove"
                | "replace_whitespace"
                | "string_char_count"
                | "nonempty_substring_count"
                | "uppercase_ascii_sum"
                | "hex_prime_count"
                | "ascii_mod_char"
                | "longest_word_length"
                | "overlapping_substring_count"
                | "spelled_number_sort"
                | "flip_case"
                | "concat_strings"
                | "filter_by_prefix"
                | "digit_substring_length_sum_count"
                | "decode_cyclic"
                | "numeric_string_parser"
                | "private_extract_first_def"
                | "private_reverse_text"
                | "private_palindrome_check"
                | "private_decode_shift_general"
                | "private_anagram_sorted"
        )
    {
        "string_transform"
    } else if digit_rotation_category(category) {
        "digit_string_boundary"
    } else if private_mbpp_collection_category(category)
        || matches!(
            category,
            "median_list"
                | "sum_list"
                | "max_list"
                | "min_list"
                | "common_elements"
                | "below_threshold"
                | "filter_integers"
                | "safe_head"
                | "list_difference"
                | "stable_negative_partition"
                | "top_k_largest"
                | "list_tail_replace"
                | "tuple_frequency_dict"
                | "tuple_item_count"
                | "count_integer_items"
                | "split_list_at_index"
                | "tuple_elementwise_division"
                | "tuple_elementwise_max"
                | "tuple_nested_elementwise_max"
                | "insert_before_each"
                | "list_chunks_every_n"
                | "combinations_with_replacement"
                | "sorted_unique_values"
                | "sort_even_index_values"
                | "increment_each_item"
                | "two_sum_zero_exists"
                | "three_sum_zero_exists"
                | "pluck_smallest_even"
                | "frequency_at_least_value"
                | "alternating_min_max_sort"
                | "palindrome_list_weight"
                | "smallest_palindrome_changes"
                | "total_match_lengths"
                | "sort_by_second"
                | "rescale_to_unit"
                | "max_tuple_difference"
                | "nested_flat_sum"
                | "positive_count"
                | "positive_filter"
                | "sublist_contains"
                | "equal_tuple_lengths"
                | "sort_list"
                | "same_pattern_sequence"
                | "tuple_all_divisible"
                | "frequency_dict"
                | "substring_in_list"
                | "closest_pair_sorted"
                | "unique_once_stable"
                | "sort_indices_multiple_three"
                | "private_max_item"
                | "private_median_even_odd"
                | "private_all_below_threshold"
                | "private_same_char_set"
                | "interval_merge_private"
                | "top_k_frequency_private"
                | "graph_reachable_private"
        )
    {
        "collection_transform"
    } else if private_mbpp_numeric_category(category)
        || matches!(
            category,
            "opposite_signs"
                | "add_numbers"
                | "abs_diff"
                | "even_number"
                | "min_three"
                | "cube_volume"
                | "cube_lateral_surface_area"
                | "cylinder_lateral_surface_area"
                | "swap_pair"
                | "count_primes_below"
                | "next_perfect_square"
                | "harmonic_sum"
                | "triangle_area_product"
                | "triangle_area_sides"
                | "count_digit_under_divisibility"
                | "divisible_by_11"
                | "multiply_three_primes"
                | "simple_power"
                | "cube_number"
                | "private_opposite_signs"
                | "private_pair_sum"
                | "private_gcd_pair_loop"
                | "private_prime_loop"
                | "base_digits"
                | "woodall_number_check"
                | "polygonal_octagonal_number"
                | "polygonal_tetrahedral_number"
                | "polygonal_centered_hexagonal_number"
                | "sphere_volume"
                | "sphere_surface_area"
                | "difference_of_squares_check"
                | "odd_length_check"
                | "closest_smaller_number"
                | "car_race_collision_count"
                | "bell_number_sequence"
                | "newman_conway_sequence"
                | "prime_fib_sequence"
                | "polynomial_zero_bisection"
                | "prime_factors"
                | "private_base_digits"
                | "private_base_digits_alt"
        )
    {
        "scalar_numeric"
    } else if matches!(category, "index_or_minus_one" | "private_index_or_default") {
        "membership_lookup"
    } else {
        "general"
    }
}

pub(super) fn category_expected_arg_count(task: &CodeTask) -> Option<usize> {
    if matches!(task.category.as_str(), "bpg_lcs_length") {
        return Some(2);
    }
    if let Some(hint) = decoder_visible_arg_count_hint(task) {
        return Some(hint);
    }
    let visible = visible_signature_arg_names(task).len();
    if visible > 0 {
        return Some(visible);
    }
    let category = task.category.as_str();
    if matches!(category, "min_three" | "triangle_area_sides") {
        return Some(3);
    }
    if matches!(category, "dict_merge_three") {
        return Some(3);
    }
    if matches!(
        category,
        "private_exec_archive_config_zip"
            | "private_exec_csv_command_outputs"
            | "private_exec_log_backup_tar"
            | "private_exec_json_extract_field"
    ) {
        return Some(2);
    }
    if matches!(category, "private_exec_system_info_dict") {
        return Some(0);
    }
    if matches!(
        category,
        "opposite_signs"
            | "add_numbers"
            | "abs_diff"
            | "replace_whitespace"
            | "cylinder_lateral_surface_area"
            | "stable_negative_partition"
            | "top_k_largest"
            | "tuple_item_count"
            | "split_list_at_index"
            | "swap_pair"
            | "tuple_elementwise_division"
            | "tuple_elementwise_max"
            | "tuple_nested_elementwise_max"
            | "insert_before_each"
            | "list_chunks_every_n"
            | "combinations_with_replacement"
            | "substring_count"
            | "safe_head"
            | "dict_required_keys"
            | "common_elements"
            | "list_difference"
            | "dot_product"
            | "clamp_number"
            | "index_or_minus_one"
            | "below_threshold"
            | "gcd_pair"
            | "is_anagram"
            | "base_digits"
            | "triangle_area_sides"
            | "modular_power_two"
            | "rotate_sequence"
            | "circular_digit_shift"
            | "palindrome_list_weight"
            | "total_match_lengths"
            | "simple_power"
            | "digit_rotate_right_private"
            | "signed_digit_rotate_private"
            | "multi_step_digit_shift_private"
            | "private_index_or_default"
            | "private_all_below_threshold"
            | "private_opposite_signs"
            | "private_decode_shift_general"
            | "private_pair_sum"
            | "private_same_char_set"
            | "private_gcd_pair_loop"
            | "private_anagram_sorted"
            | "private_base_digits"
            | "private_base_digits_alt"
            | "sublist_contains"
            | "same_pattern_sequence"
            | "tuple_all_divisible"
            | "substring_in_list"
            | "overlapping_substring_count"
            | "filter_by_prefix"
    ) || digit_rotation_category(category)
    {
        Some(2)
    } else {
        Some(1)
    }
}

pub(super) fn decoder_visible_arg_count_hint(task: &CodeTask) -> Option<usize> {
    task.raw
        .get("decoder_contract")
        .and_then(|contract| contract.get("visible_arg_count_hint"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

pub(super) fn body_mentions_token(body: &str, token: &str) -> bool {
    tokenize_code(body).iter().any(|item| item == token)
}
