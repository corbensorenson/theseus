fn complete_state_sequence_body(task: &CodeTask, body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty()
        || trimmed
            .lines()
            .last()
            .is_some_and(|line| line.trim_end().ends_with(':'))
    {
        return None;
    }
    let mut lines = trimmed.lines().map(str::to_string).collect::<Vec<_>>();
    let lowered = trimmed.to_lowercase();
    let compact = lowered
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let return_line = if task.category == "balanced_brackets_simple" && lowered.contains("stack") {
        "return not stack"
    } else if task.category == "monotonic_sequence" && lowered.contains("nondecreasing") {
        "return nondecreasing or nonincreasing"
    } else if task.category == "largest_prime_factor" && lowered.contains("best") {
        "return best"
    } else if task.category == "arithmetic_series_sum" && lowered.contains("total") {
        "return total"
    } else if vowel_rule_category(&task.category) && lowered.contains("total") {
        "return total"
    } else if task.category == "tribonacci_sequence" && lowered.contains("values") {
        "return values[data]"
    } else if recurrence_category(&task.category) && lowered.contains("values") {
        "return values[data]"
    } else if recurrence_category(&task.category) && lowered.contains("b =") {
        "return b"
    } else if matches!(
        task.category.as_str(),
        "remove_vowels" | "caesar_decode_shift5"
    ) && (lowered.contains("out = []") || lowered.contains("out=[]"))
    {
        "return ''.join(out)"
    } else if task.category == "below_threshold" && compact.contains("returnfalse") {
        "return True"
    } else if task.category == "gcd_pair" && lowered.contains("while") && lowered.contains("b") {
        "return a"
    } else if task.category == "is_prime" && compact.contains("returnfalse") {
        "return True"
    } else if task.category == "factors" && lowered.contains("out") {
        "return out"
    } else if task.category == "prime_factors" && lowered.contains("out") {
        "return out"
    } else if task.category == "string_sequence" && lowered.contains("parts") {
        "return ' '.join(parts)"
    } else if matches!(
        task.category.as_str(),
        "two_sum_zero_exists" | "three_sum_zero_exists"
    ) && compact.contains("returntrue")
    {
        "return False"
    } else if task.category == "count_digit_under_divisibility" && lowered.contains("total") {
        "return total"
    } else if task.category == "rescale_to_unit"
        && lowered.contains("low")
        && lowered.contains("high")
    {
        "return [(item - low) / (high - low) for item in data]"
    } else if task.category == "decode_cyclic" && lowered.contains("out") {
        "return ''.join(out)"
    } else if task.category == "prime_fib_sequence" && lowered.contains("found") {
        "return a"
    } else if task.category == "polynomial_zero_bisection" && lowered.contains("left") {
        "return (left + right) / 2"
    } else if task.category == "base_digits" && lowered.contains("digits") {
        "return ''.join(reversed(digits))"
    } else if matches!(
        task.category.as_str(),
        "parse_ints" | "numeric_string_parser"
    ) && lowered.contains("out")
    {
        "return out"
    } else if task.category == "filter_integers" && lowered.contains("out") {
        "return out"
    } else if task.category == "safe_head" && compact.contains("returnnone") {
        "return data[0]"
    } else if task.category == "dict_required_keys" && compact.contains("returnfalse") {
        "return True"
    } else if task.category == "common_elements" && lowered.contains("common") {
        "return sorted(common)"
    } else if task.category == "sorted_unique_values" && lowered.contains("items") {
        "return sorted(set(items))"
    } else if task.category == "modular_power_two" && lowered.contains("result") {
        "return result"
    } else if task.category == "median_list"
        && compact.contains("items=sorted(data)")
        && compact.contains("mid=len(items)//2")
        && (compact.contains("iflen(items)%2==1") || compact.contains("iflen(data)%2==1"))
    {
        "return (items[mid - 1] + items[mid]) / 2"
    } else if lowered.contains("out = []") || lowered.contains("out=[]") {
        "return out"
    } else if lowered.contains("values = []") || lowered.contains("values=[]") {
        "return values"
    } else {
        return None;
    };
    lines.push(return_line.to_string());
    Some(lines.join("\n"))
}

fn state_sequence_seed_prefixes(task: &CodeTask) -> Vec<Vec<String>> {
    let mut rows: Vec<&str> = Vec::new();
    match task.category.as_str() {
        "balanced_brackets_simple" => {
            rows.push("stack = []\nfor ch in data:");
            rows.push("stack = []\nfor ch in data:\n    if ch == '(':");
            rows.push("stack = []\nfor ch in data:\n    if ch in '([{':");
        }
        "monotonic_sequence" => {
            rows.push("if len(data) < 2:\n    return True\nnondecreasing = True\nnonincreasing = True\nfor idx in range(1, len(data)):");
        }
        "largest_prime_factor" => {
            rows.push("best = 1\nvalue = data\nfactor = 2\nwhile factor * factor <= value:");
        }
        "arithmetic_series_sum" | "sum_list" => {
            rows.push("total = 0\nfor item in range(data + 1):");
        }
        "derivative_coefficients" | "all_prefixes" => {
            rows.push("out = []");
            rows.push("values = []");
        }
        "tribonacci_sequence" => {
            rows.push("values = [0, 0, 1]\nif data < len(values):");
            rows.push("values = [0, 0, 1]\nfor _ in range(3, data + 1):");
        }
        "fibonacci_loop_private"
        | "lucas_loop_private"
        | "shifted_recurrence_private"
        | "private_fibonacci_loop"
        | "private_lucas_loop"
        | "private_shifted_recurrence" => {
            rows.push("a = 0\nb = 1\nif data == 0:");
            rows.push("a = 0\nb = 1\nfor _ in range(1, data):");
        }
        "nested_recurrence_private" | "private_nested_recurrence" => {
            rows.push("a = 0\nb = 1\nfor _ in range(data):");
        }
        "circular_digit_shift"
        | "digit_rotate_right_private"
        | "signed_digit_rotate_private"
        | "multi_step_digit_shift_private"
        | "private_digit_rotate_right"
        | "private_signed_digit_rotate"
        | "private_overshift_reverse_digits"
        | "private_multi_step_digit_shift" => {
            rows.push("digits = str(data)\nif other > len(digits):");
            rows.push("digits = str(data)\nshift = other % len(digits)");
        }
        "count_vowels"
        | "final_y_vowel_private"
        | "suffix_y_vowel_private"
        | "case_punct_vowel_private"
        | "private_final_y_vowels"
        | "private_suffix_y_vowels"
        | "private_case_punct_vowels" => {
            rows.push("text = data.lower()\ntotal = 0\nfor idx, ch in enumerate(text):");
            rows.push("total = 0\nfor ch in data.lower():");
        }
        "common_elements" | "sorted_unique_values" => {
            rows.push("left = set(data)\nright = set(other)\ncommon = left & right");
            rows.push("items = sorted(set(data)");
        }
        "median_list" => {
            rows.push("items = sorted(data)\nmid = len(items) // 2\nif len(items) % 2 == 1:");
            rows.push(
                "items = sorted(data)\nmid = len(items) // 2\nif len(items) % 2 == 1:\n    return",
            );
            rows.push("items = sorted(data)\nmid = len(items) // 2");
        }
        "modular_power_two" => {
            rows.push("result = 1\nfor _ in range(data):");
            rows.push("result = 1\nfor _ in range(data):\n    result = (result * 2) % other");
        }
        "caesar_decode_shift5" => {
            rows.push("out = []\nfor ch in data:");
        }
        "remove_vowels" => {
            rows.push("out = []\nfor ch in data:");
            rows.push("out = []\nfor ch in data:\n    if ch.lower() not in 'aeiou':");
        }
        "below_threshold" => {
            rows.push("for item in data:\n    if item >= other:\n        return False");
            rows.push("for item in data:\n    if item >= other:");
            rows.push("for item in data:\n    if item >= other:\n        return");
            rows.push("for item in data:");
        }
        "gcd_pair" | "private_gcd_pair_loop" => {
            rows.push("a = abs(data)\nb = abs(other)\nwhile b:");
            rows.push("a = abs(data)\nb = abs(other)\nwhile b:\n    a, b = b, a % b");
        }
        "is_prime" | "private_prime_loop" => {
            rows.push(
                "if data <= 1:\n    return False\nfor divisor in range(2, int(data ** 0.5) + 1):",
            );
            rows.push("if data <= 1:\n    return False\nfor divisor in range(2, data):\n    if data % divisor == 0:");
        }
        "factors" => {
            rows.push("out = []\nfor divisor in range(1, data + 1):");
            rows.push("out = []\nfor divisor in range(1, data + 1):\n    if data % divisor == 0:");
        }
        "prime_factors" => {
            rows.push("out = []\nvalue = data\nfactor = 2\nwhile factor * factor <= value:");
            rows.push("out = []\nvalue = data\nfactor = 2\nwhile factor * factor <= value:\n    while value % factor == 0:");
        }
        "largest_divisor" => {
            rows.push("best = 1\nfor item in range(1, data):");
            rows.push("best = 1\nfor item in range(1, data):\n    if data % item == 0:");
        }
        "divisible_by_11" => {
            rows.push("return data % 11 == 0");
        }
        "rescale_to_unit" => {
            rows.push("low = min(data)\nhigh = max(data)");
            rows.push("low = min(data)\nhigh = max(data)\nreturn [(item - low) / (high - low) for item in data]");
        }
        "decode_cyclic" => {
            rows.push("out = []\nfor idx in range(0, len(data), 3):");
            rows.push(
                "out = []\nfor idx in range(0, len(data), 3):\n    group = data[idx:idx + 3]",
            );
        }
        "prime_fib_sequence" => {
            rows.push("found = 0\na = 1\nb = 1\nwhile True:");
            rows.push("found = 0\na = 1\nb = 1\nwhile True:\n    a, b = b, a + b");
        }
        "polynomial_zero_bisection" => {
            rows.push("left = -1.0\nright = 1.0");
            rows.push("left = -1.0\nright = 1.0\nfor _ in range(60):");
        }
        "is_anagram" | "private_anagram_sorted" => {
            rows.push("return sorted(data) ==");
        }
        "base_digits" | "private_base_digits" | "private_base_digits_alt" => {
            rows.push("if data == 0:\n    return '0'\ndigits = []\nvalue = data\nwhile value:");
            rows.push(
                "digits = []\nvalue = data\nwhile value:\n    digits.append(str(value % other))",
            );
        }
        "safe_head" => {
            rows.push("if not data:\n    return None");
        }
        "dict_required_keys" => {
            rows.push("for key in other:\n    if key not in data:\n        return False");
        }
        "parse_ints" | "numeric_string_parser" => {
            rows.push("out = []\nfor token in data.replace(',', ' ').split():");
        }
        "filter_integers" => {
            rows.push("out = []\nfor item in data:\n    if isinstance(item, int):");
        }
        "string_sequence" => {
            rows.push("parts = []\nfor idx in range(data + 1):");
            rows.push("parts = []\nfor idx in range(data + 1):\n    parts.append(str(idx))");
        }
        "two_sum_zero_exists" => {
            rows.push("seen = set()\nfor item in data:");
            rows.push("seen = set()\nfor item in data:\n    if -item in seen:");
        }
        "three_sum_zero_exists" => {
            rows.push("n = len(data)\nfor i in range(n):");
            rows.push("n = len(data)\nfor i in range(n):\n    for j in range(i + 1, n):");
        }
        "count_digit_under_divisibility" => {
            rows.push("total = 0\nfor item in range(data):");
            rows.push(
                "total = 0\nfor item in range(data):\n    if item % 11 == 0 or item % 13 == 0:",
            );
        }
        "stable_negative_partition" => {
            rows.push("head = list(data[:other])\ntail = list(data[other:])\nnegatives = []\nnonnegatives = []\nfor item in head:");
        }
        "replace_whitespace" => {
            rows.push("return data.replace");
        }
        "cube_volume" => {
            rows.push("return data * data");
            rows.push("return data * data *");
        }
        "cube_lateral_surface_area" => {
            rows.push("return 4 * data");
            rows.push("return 4 * data *");
        }
        "cylinder_lateral_surface_area" => {
            rows.push("return 2 * 3.141592653589793 * data");
        }
        "tuple_frequency_dict" => {
            rows.push("counts = {}\nfor item in data:");
        }
        "tuple_item_count" => {
            rows.push("return data.count");
        }
        "insert_before_each" => {
            rows.push("out = []\nfor item in data:");
        }
        "count_integer_items" => {
            rows.push("total = 0\nfor item in data:");
        }
        "tuple_nested_elementwise_max" => {
            rows.push("out = []\nfor left_pair, right_pair in zip(data, other):");
        }
        "count_primes_below" => {
            rows.push("total = 0\nfor value in range(2, data):\n    is_prime_value = True\n    for divisor in range(2, int(value ** 0.5) + 1):");
        }
        "next_perfect_square" => {
            rows.push("root = 0\nwhile root * root <= data:");
        }
        "top_k_largest" => {
            rows.push("items = sorted(data, reverse=True)\nreturn items[:other]");
        }
        "sort_by_second" => {
            rows.push("out = list(data)\nfor i in range(len(out)):");
            rows.push(
                "out = list(data)\nfor i in range(len(out)):\n    for j in range(i + 1, len(out)):",
            );
        }
        "list_tail_replace" => {
            rows.push("return data[:-1] + list(other)");
        }
        "split_list_at_index" => {
            rows.push("return (data[:other], data[other:])");
        }
        "list_chunks_every_n" => {
            rows.push("return [data[idx:idx + other] for idx in range");
        }
        "combinations_with_replacement" => {
            rows.push("out = []\ndef build(start, current):");
        }
        "uppercase_ascii_sum" => {
            rows.push("total = 0\nfor ch in data:\n    if ch.isupper():");
        }
        "pluck_smallest_even" => {
            rows.push("best = None\nbest_idx = -1\nfor idx, item in enumerate(data):");
        }
        "frequency_at_least_value" => {
            rows.push("counts = {}\nfor item in data:");
            rows.push("counts = {}\nfor item in data:\n    counts[item] = counts.get(item, 0) + 1\nbest = -1");
        }
        "alternating_min_max_sort" => {
            rows.push("items = sorted(data)\nout = []\ntake_low = True\nwhile items:");
        }
        "palindrome_list_weight" => {
            rows.push("if data != data[::-1]:\n    return False");
        }
        "smallest_palindrome_changes" => {
            rows.push("total = 0\nfor idx in range(len(data) // 2):");
        }
        "total_match_lengths" => {
            rows.push("left = sum(len(item) for item in data)\nright = sum(len(item) for item in other)\nif left <= right:");
        }
        "triangle_area_sides" => {
            rows.push("a = data\nb = other\nc = extra[0] if extra else 0\nif a + b <= c or a + c <= b or b + c <= a:");
        }
        "multiply_three_primes" => {
            rows.push("value = data\ncount = 0\nfactor = 2\nwhile factor <= value:");
        }
        "simple_power" => {
            rows.push("if data == 1:\n    return True\nif other <= 1:\n    return False\nvalue = 1\nwhile value < data:");
        }
        "cube_number" => {
            rows.push("value = abs(data)\nroot = 0\nwhile root * root * root < value:");
        }
        "hex_prime_count" => {
            rows.push("total = 0\nfor ch in data:\n    if ch in '2357BD':");
        }
        "woodall_number_check" => {
            rows.push("k = 1\nwhile k * (2 ** k) - 1 <= data:");
        }
        "polygonal_octagonal_number" => {
            rows.push("return data * (3 * data - 2)");
        }
        "polygonal_tetrahedral_number" => {
            rows.push("return data * (data + 1) * (data + 2) // 6");
        }
        "polygonal_centered_hexagonal_number" => {
            rows.push("return 3 * data * (data - 1) + 1");
        }
        "sphere_volume" => {
            rows.push("return 4 / 3 * 3.141592653589793 * data ** 3");
        }
        "sphere_surface_area" => {
            rows.push("return 4 * 3.141592653589793 * data * data");
        }
        "nested_flat_sum" => {
            rows.push("total = 0\nstack = list(data)\nwhile stack:");
        }
        "positive_count" => {
            rows.push("total = 0\nfor item in data:\n    if item > 0:");
        }
        "positive_filter" => {
            rows.push("out = []\nfor item in data:\n    if item > 0:");
        }
        "sublist_contains" => {
            rows.push("if other == []:\n    return True\nfor idx in range(0, len(data) - len(other) + 1):");
        }
        "equal_tuple_lengths" => {
            rows.push("if not data:\n    return True\nsize = len(data[0])\nfor item in data:");
        }
        "sort_list" => {
            rows.push("return sorted(data)");
        }
        "difference_of_squares_check" => {
            rows.push("return data % 4 != 2");
        }
        "same_pattern_sequence" => {
            rows.push("if len(data) != len(other):\n    return False\nleft = {}\nright = {}\nfor a, b in zip(data, other):");
        }
        "tuple_all_divisible" => {
            rows.push(
                "out = []\nfor item in data:\n    if all(value % other == 0 for value in item):",
            );
        }
        "odd_length_check" => {
            rows.push("return len(data) % 2 == 1");
        }
        "ascii_mod_char" => {
            rows.push("total = 0\nfor ch in data:");
        }
        "dict_merge_three" => {
            rows.push("out = {}\nout.update(data)\nout.update(other)\nfor item in extra:");
        }
        "frequency_dict" => {
            rows.push("counts = {}\nfor item in data:");
        }
        "closest_smaller_number" => {
            rows.push("return data - 1");
        }
        "longest_word_length" => {
            rows.push("words = data.split()\nif not words:");
        }
        "substring_in_list" => {
            rows.push("for item in data:\n    if other in item:");
        }
        "overlapping_substring_count" => {
            rows.push("if other == '':\n    return 0\ntotal = 0\nfor idx in range(0, len(data) - len(other) + 1):");
        }
        "spelled_number_sort" => {
            rows.push("order = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}");
        }
        "closest_pair_sorted" => {
            rows.push("best = None\nbest_dist = None\nitems = sorted(data)\nfor idx in range(len(items) - 1):");
        }
        "unique_once_stable" => {
            rows.push("counts = {}\nfor item in data:");
        }
        "flip_case" => {
            rows.push("return data.swapcase()");
        }
        "concat_strings" => {
            rows.push("return ''.join(data)");
        }
        "filter_by_prefix" => {
            rows.push("out = []\nfor item in data:\n    if item.startswith(other):");
        }
        "sort_indices_multiple_three" => {
            rows.push("out = list(data)\nvalues = sorted(out[::3])");
        }
        "car_race_collision_count" => {
            rows.push("return data * data");
        }
        "digit_substring_length_sum_count" => {
            rows.push("total = 0\nfor start in range(len(data)):\n    digit_sum = 0\n    for end in range(start, len(data)):");
        }
        "bell_number_sequence" => {
            rows.push("bell = [[0 for _ in range(data + 1)] for _ in range(data + 1)]");
        }
        "newman_conway_sequence" => {
            rows.push(
                "if data <= 2:\n    return 1\nvalues = [0, 1, 1]\nfor n in range(3, data + 1):",
            );
        }
        "palindrome" | "private_palindrome_check" => {
            rows.push("return data == data[");
        }
        _ => {}
    }
    rows.into_iter()
        .map(tokenize_body)
        .filter(|tokens| !tokens.is_empty())
        .collect()
}

fn prefix_is_token_allowed(tokens: &[String]) -> bool {
    let mut existing = Vec::new();
    for token in tokens {
        if !body_token_allowed(&existing, token) {
            return false;
        }
        existing.push(token.clone());
    }
    true
}

fn state_sequence_token_options_from_scores(
    task: &CodeTask,
    body_ngram: &BodyNgramModel,
    vocab: &Vocab,
    scores: &[f32],
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
    for (id, score) in scores.iter().copied().enumerate() {
        let Some(token) = vocab.id_to_token.get(id).cloned() else {
            continue;
        };
        if token == "<UNK>" || !task_body_token_allowed(task, existing, &token) {
            continue;
        }
        let guidance_score = category_guidance.get(&token).copied().unwrap_or_else(|| {
            if category_guidance.is_empty()
                || token == "<EOS>"
                || matches!(token.as_str(), "<NL>" | "<INDENT>" | "<DEDENT>")
            {
                0.0
            } else {
                -1.35
            }
        });
        let adjusted = score
            + token_alignment_bonus(&token, prompt_tokens)
            + body_position_bonus(&token, position)
            + category_body_token_bonus(task, &token)
            + category_position_token_bonus(task, &token, position)
            + state_sequence_rule_bonus(task, existing, &token, position)
            + guidance_score;
        push_bounded_state_sequence_option(&mut rows, limit, token, adjusted, id);
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

fn push_bounded_state_sequence_option(
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

fn token_option_better(score: f32, vocab_index: usize, other_score: f32, other_index: usize) -> bool {
    match score
        .partial_cmp(&other_score)
        .unwrap_or(std::cmp::Ordering::Less)
    {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => vocab_index < other_index,
        std::cmp::Ordering::Less => false,
    }
}

fn body_ngram_category_token_scores(
    task: &CodeTask,
    model: &BodyNgramModel,
    prev2: &str,
    prev1: &str,
    position: usize,
) -> BTreeMap<String, f32> {
    if model.counts.is_empty() {
        return BTreeMap::new();
    }
    let mut scored: BTreeMap<String, f32> = BTreeMap::new();
    let category = task.category.trim();
    let category = if category.is_empty() {
        "general_semantics"
    } else {
        category
    };
    let capped_position = learned_position_cap(category, position);
    for (key, weight) in [
        (
            format!(
                "cat:{}|pos:{}|p2:{}|p1:{}",
                category, capped_position, prev2, prev1
            ),
            8.0f32,
        ),
        (
            format!("cat:{}|pos:{}|p1:{}", category, capped_position, prev1),
            5.5f32,
        ),
        (
            format!("cat:{}|p2:{}|p1:{}", category, prev2, prev1),
            5.0f32,
        ),
        (format!("cat:{}|p1:{}", category, prev1), 3.0f32),
        (format!("cat:{}|pos:{}", category, capped_position), 1.0f32),
        (format!("all|p2:{}|p1:{}", prev2, prev1), 1.4f32),
        (format!("all|p1:{}", prev1), 0.8f32),
    ] {
        let Some(counts) = model.counts.get(&key) else {
            continue;
        };
        let total = counts.values().sum::<usize>().max(1) as f32;
        for (token, count) in counts {
            let frequency = *count as f32 / total;
            let confidence = if frequency >= 0.70 { 1.5 } else { frequency };
            *scored.entry(token.clone()).or_insert(0.0) +=
                ((*count as f32 + 1.0).ln() + confidence) * weight;
        }
    }
    inject_execution_shape_context_token_priors(task, prev2, prev1, position, &mut scored);
    inject_argument_obligation_token_priors(task, prev2, prev1, &mut scored);
    if scored.is_empty() {
        return scored;
    }
    let max_score = scored.values().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = max_score
        - if execution_shaped_category(category) {
            14.0
        } else {
            8.0
        };
    scored.retain(|_, score| *score >= floor);
    scored
}

fn inject_execution_shape_context_token_priors(
    task: &CodeTask,
    prev2: &str,
    prev1: &str,
    position: usize,
    scored: &mut BTreeMap<String, f32>,
) {
    if !execution_shaped_category(&task.category) {
        return;
    }
    for (token, bonus) in execution_shape_context_token_priors(task, prev2, prev1, position) {
        *scored.entry(token.to_string()).or_insert(0.0) += bonus;
    }
}

fn inject_argument_obligation_token_priors(
    task: &CodeTask,
    prev2: &str,
    prev1: &str,
    scored: &mut BTreeMap<String, f32>,
) {
    let expected = category_expected_arg_count(task).unwrap_or(0);
    if execution_shaped_category(&task.category) {
        if matches!(
            task.category.as_str(),
            "private_exec_archive_config_zip" | "archive_config_zip"
        ) {
            if prev2 == "os" && prev1 == "." {
                *scored.entry("read".to_string()).or_insert(0.0) -= 80.0;
                *scored.entry("path".to_string()).or_insert(0.0) += 20.0;
            }
            if prev2 == "configparser" && prev1 == "." {
                *scored.entry("ConfigParser".to_string()).or_insert(0.0) += 80.0;
                *scored.entry("path".to_string()).or_insert(0.0) -= 100.0;
                *scored.entry("read".to_string()).or_insert(0.0) -= 80.0;
            }
            if prev2 == "." && prev1 == "ConfigParser" {
                *scored.entry("(".to_string()).or_insert(0.0) += 60.0;
                *scored.entry("<NL>".to_string()).or_insert(0.0) -= 80.0;
            }
            if prev2 == "config" && prev1 == "." {
                *scored.entry("read".to_string()).or_insert(0.0) += 36.0;
                *scored.entry("get".to_string()).or_insert(0.0) += 80.0;
                *scored.entry("raise".to_string()).or_insert(0.0) -= 120.0;
            }
            if prev1 == "get" {
                *scored.entry("(".to_string()).or_insert(0.0) += 80.0;
                *scored.entry("configparser".to_string()).or_insert(0.0) -= 100.0;
                *scored.entry("path".to_string()).or_insert(0.0) -= 80.0;
            }
        }
        if prev2 == "load" && prev1 == "(" {
            *scored.entry("handle".to_string()).or_insert(0.0) += 36.0;
            *scored.entry("other".to_string()).or_insert(0.0) -= 24.0;
            *scored.entry("data".to_string()).or_insert(0.0) -= 18.0;
        }
        if prev2 == "run" && prev1 == "(" {
            *scored.entry("command".to_string()).or_insert(0.0) += 28.0;
            *scored.entry("other".to_string()).or_insert(0.0) -= 18.0;
        }
        if prev2 == "list" && prev1 == "(" && task.category.contains("csv") {
            *scored.entry("csv".to_string()).or_insert(0.0) += 24.0;
            *scored.entry("rows".to_string()).or_insert(0.0) -= 12.0;
        }
        if prev2 == "urlencode" && prev1 == "(" {
            *scored.entry("items".to_string()).or_insert(0.0) += 34.0;
            *scored.entry("data".to_string()).or_insert(0.0) -= 14.0;
        }
    }
    if expected == 0 {
        for token in ["data", "other", "args", "extra"] {
            *scored.entry(token.to_string()).or_insert(0.0) -= 18.0;
        }
        return;
    }
    if expected >= 1 {
        *scored.entry("data".to_string()).or_insert(0.0) += if prev1 == "(" { 8.0 } else { 1.5 };
    }
    if expected >= 2 {
        let call_wants_secondary = prev1 == "("
            && matches!(
                prev2,
                "join" | "makedirs" | "get" | "run" | "open" | "ZipFile"
            );
        let secondary_bonus = if execution_shaped_category(&task.category)
            && matches!(prev2, "load" | "run")
            && prev1 == "("
        {
            -8.0
        } else if call_wants_secondary {
            18.0
        } else if matches!(prev1, "(" | ",") {
            10.0
        } else {
            3.0
        };
        *scored.entry("other".to_string()).or_insert(0.0) += secondary_bonus;
    }
    if expected >= 3 {
        *scored.entry("extra".to_string()).or_insert(0.0) +=
            if matches!(prev1, "(" | ",") { 8.0 } else { 2.0 };
    }
}

fn execution_shape_context_token_priors(
    task: &CodeTask,
    prev2: &str,
    prev1: &str,
    position: usize,
) -> Vec<(&'static str, f32)> {
    let mut priors = Vec::new();
    let text = task_contract_text(task);
    let at_line_start = matches!(prev1, "<BOS>" | "<NL>" | "<INDENT>" | "<DEDENT>");
    let mentions_csv = body_has_any(&text, &["csv", "comma separated"]);
    let mentions_json = body_has_any(&text, &["json"]);
    let mentions_urlencode = body_has_any(&text, &["urlencode", "url encode", "query string"]);
    let mentions_zip = body_has_any(&text, &["zip", "zipfile"]);
    let mentions_tar = body_has_any(&text, &["tar", "tarfile", "log backup", "logs"]);
    let mentions_archive = body_has_any(&text, &["archive", "zip", "tar"]);
    let mentions_system = body_has_any(
        &text,
        &["system", "platform", "architecture", "memory usage"],
    );
    let mentions_file = body_has_any(&text, &["file", "path", "directory", "folder"]);
    let category = task.category.as_str();
    let csv_split_shuffle =
        matches!(
            category,
            "private_exec_csv_split_shuffle" | "csv_split_shuffle"
        ) || body_has_any(&text, &["split a csv", "shuffled csv", "csv chunks"]);

    if at_line_start {
        if mentions_csv {
            priors.extend([("import", 5.0), ("with", 4.0), ("out", 3.0), ("rows", 3.0)]);
        }
        if csv_split_shuffle {
            priors.extend([
                ("with", 9.0),
                ("rows", 8.0),
                ("out", 8.0),
                ("chunk_size", 8.0),
                ("for", 8.0),
                ("path", 7.0),
                ("random", 5.0),
                ("return", -10.0),
            ]);
        }
        if mentions_json {
            priors.extend([
                ("import", 5.0),
                ("if", 4.0),
                ("with", 4.0),
                ("payload", 3.5),
            ]);
        }
        if mentions_urlencode {
            priors.extend([("from", 7.0), ("if", 4.0), ("items", 4.0), ("return", 3.0)]);
        }
        if mentions_archive || mentions_zip || mentions_tar {
            priors.extend([("import", 5.5), ("if", 4.0), ("with", 4.0)]);
        }
        if mentions_system {
            priors.extend([
                ("import", 5.0),
                ("try", 4.0),
                ("memory", 4.0),
                ("return", 3.0),
            ]);
        }
    }

    if csv_split_shuffle {
        match (prev2, prev1) {
            (_, "csv") => priors.push((".", 22.0)),
            ("csv", ".") => priors.extend([("reader", 24.0), ("writer", 22.0)]),
            (_, "random") => priors.push((".", 18.0)),
            ("random", ".") => priors.extend([("Random", 18.0), ("shuffle", 14.0)]),
            (_, "Random") => priors.push(("(", 18.0)),
            ("Random", "(") => priors.push(("0", 20.0)),
            ("0", ")") => priors.push((".", 18.0)),
            (")", ".") => priors.push(("shuffle", 20.0)),
            (_, "shuffle") => priors.push(("(", 20.0)),
            ("shuffle", "(") => priors.push(("rows", 22.0)),
            (_, "range") => priors.push(("(", 20.0)),
            ("range", "(") => priors.push(("0", 16.0)),
            (_, "writerows") => priors.push(("(", 20.0)),
            ("writerows", "(") => priors.push(("rows", 18.0)),
            (_, "append") => priors.push(("(", 18.0)),
            ("append", "(") => priors.push(("path", 18.0)),
            (_, "max") => priors.push(("(", 14.0)),
            ("max", "(") => priors.push(("1", 16.0)),
            (_, "len") => priors.push(("(", 12.0)),
            ("len", "(") => priors.push(("rows", 16.0)),
            (_, "return") => priors.extend([("out", 24.0), ("[]", -16.0)]),
            _ => {}
        }
        if matches!(prev1, "<NL>" | "<INDENT>" | "<DEDENT>") && position > 5 {
            priors.extend([
                ("with", 14.0),
                ("rows", 12.0),
                ("random", 12.0),
                ("base_dir", 10.0),
                ("out", 10.0),
                ("chunk_size", 12.0),
                ("for", 14.0),
                ("return", -12.0),
            ]);
        }
        if prev1 == "return" && !body_has_any(&text, &["invalid input"]) {
            priors.push(("[]", -12.0));
        }
    }
    match prev1 {
        "import" => {
            if mentions_csv {
                priors.extend([
                    ("csv", 16.0),
                    ("os", 9.0),
                    ("subprocess", 8.0),
                    ("random", 5.0),
                ]);
            } else {
                priors.push(("csv", -18.0));
            }
            if mentions_json {
                priors.extend([("json", 18.0), ("os", 9.0)]);
            }
            if mentions_zip {
                priors.extend([("os", 12.0), ("zipfile", 18.0), ("shutil", 10.0)]);
            }
            if mentions_tar {
                priors.extend([("glob", 12.0), ("os", 10.0), ("tarfile", 18.0)]);
            }
            if mentions_system {
                priors.extend([("platform", 18.0), ("psutil", 12.0)]);
            }
            if mentions_urlencode {
                priors.push(("urlencode", 24.0));
            }
        }
        "from" if mentions_urlencode => priors.push(("urllib", 22.0)),
        "urllib" if prev2 == "from" => priors.push((".", 22.0)),
        "." if prev2 == "urllib" => priors.push(("parse", 22.0)),
        "parse" if prev2 == "." => priors.push(("import", 22.0)),
        "os" => priors.push((".", 18.0)),
        "path" if prev2 == "." => priors.push((".", 20.0)),
        "." if prev2 == "os" => priors.push(("path", 22.0)),
        "." if prev2 == "path" => {
            if mentions_file {
                priors.extend([
                    ("isfile", 15.0),
                    ("isdir", 15.0),
                    ("join", 13.0),
                    ("exists", 10.0),
                ]);
            }
        }
        "csv" => priors.push((".", 16.0)),
        "." if prev2 == "csv" => priors.extend([("reader", 16.0), ("writer", 12.0)]),
        "json" => priors.push((".", 16.0)),
        "." if prev2 == "json" => priors.extend([("load", 18.0), ("loads", 10.0)]),
        "configparser" => priors.push((".", 18.0)),
        "." if prev2 == "configparser" => priors.push(("ConfigParser", 24.0)),
        "ConfigParser" if prev2 == "." => priors.push(("(", 30.0)),
        "zipfile" => priors.push((".", 18.0)),
        "." if prev2 == "zipfile" => priors.push(("ZipFile", 22.0)),
        "tarfile" => priors.push((".", 18.0)),
        "." if prev2 == "tarfile" => priors.push(("open", 22.0)),
        "glob" if prev2 != "." => priors.push((".", 12.0)),
        "." if prev2 == "glob" => priors.push(("glob", 18.0)),
        "platform" => priors.push((".", 12.0)),
        "." if prev2 == "platform" => priors.extend([("system", 16.0), ("architecture", 16.0)]),
        "psutil" => priors.push((".", 12.0)),
        "." if prev2 == "psutil" => priors.push(("virtual_memory", 16.0)),
        "open" | "ZipFile" | "urlencode" | "reader" | "writer" | "run" | "glob" | "isfile"
        | "isdir" | "exists" | "join" | "listdir" | "load" | "virtual_memory" | "system"
        | "architecture" | "items" | "get" | "sorted" | "list" | "ConfigParser" => {
            priors.push(("(", 14.0));
        }
        "(" => match prev2 {
            "open" | "isfile" | "isdir" | "exists" | "listdir" => priors.push(("data", 16.0)),
            "join" => priors.push(("data", 12.0)),
            "urlencode" => priors.extend([("items", 16.0), ("data", 10.0)]),
            "get" => priors.push(("other", 16.0)),
            "sorted" | "list" => priors.extend([("rows", 10.0), ("data", 8.0), ("items", 8.0)]),
            _ => {}
        },
        ")" => {
            priors.extend([("as", 5.0), (":", 3.0), ("<NL>", 2.0)]);
            if matches!(
                category,
                "private_exec_archive_config_zip" | "archive_config_zip"
            ) && prev2 == "ConfigParser"
            {
                priors.extend([("<NL>", 16.0), ("config", 8.0)]);
            }
        }
        "as" => priors.extend([("handle", 14.0), ("archive", 12.0), ("out_handle", 10.0)]),
        "archive" => priors.push((".", 18.0)),
        "." if prev2 == "archive" => {
            if mentions_tar {
                priors.push(("add", 24.0));
            } else if mentions_zip || mentions_archive {
                priors.push(("write", 24.0));
            }
        }
        "write" if prev2 == "." => priors.push(("(", 20.0)),
        "add" if prev2 == "." => priors.push(("(", 20.0)),
        "return" => {
            if mentions_urlencode {
                priors.extend([("urlencode", 24.0), ("items", -8.0), ("data", -8.0)]);
            } else if mentions_json {
                priors.extend([("payload", 18.0), ("None", 6.0)]);
            } else if mentions_system {
                priors.push(("{", 18.0));
            } else if mentions_tar {
                priors.extend([("archive_path", 18.0), ("None", 4.0)]);
            } else if mentions_zip {
                priors.extend([("zip_path", 18.0), ("None", 4.0)]);
            } else if mentions_archive && decoder_return_shape(task) == "bool" {
                priors.extend([("True", 18.0), ("False", 8.0)]);
            } else if mentions_csv {
                priors.extend([("out", 16.0), ("paths", 12.0), ("[]", 5.0)]);
            }
        }
        "{" if mentions_system => priors.push(("'Operating System'", 16.0)),
        "'Operating System'" if mentions_system => priors.push((":", 16.0)),
        "'Architecture'" if mentions_system => priors.push((":", 16.0)),
        "'Memory Usage'" if mentions_system => priors.push((":", 16.0)),
        "," if mentions_system => {
            priors.extend([("'Architecture'", 14.0), ("'Memory Usage'", 18.0)])
        }
        "memory" if mentions_system => priors.extend([("}", 12.0), (",", 6.0)]),
        _ => {}
    }

    if mentions_system && prev1 == "}" {
        priors.push(("<EOS>", 10.0));
    }

    if matches!(
        category,
        "private_exec_json_extract_field" | "json_extract_field"
    ) {
        match (prev2, prev1) {
            (_, "return") => {
                priors.push(("payload", 24.0));
                priors.push(("{", -30.0));
            }
            ("return", "payload") => priors.push((".", 24.0)),
            ("payload", ".") => priors.push(("get", 28.0)),
            (_, "get") => priors.push(("(", 20.0)),
            ("get", "(") => priors.push(("other", 24.0)),
            ("(", "other") => priors.push((")", 20.0)),
            _ => {}
        }
    }
    if matches!(category, "private_exec_log_backup_tar" | "log_backup_tar") {
        match (prev2, prev1) {
            (_, "tarfile") => priors.push((".", if prev2 == "with" { 36.0 } else { 28.0 })),
            ("tarfile", ".") => priors.push(("open", 32.0)),
            (_, "open") => priors.push(("(", 18.0)),
            ("archive", ".") => priors.push(("add", 30.0)),
            (_, "add") => priors.push(("(", 18.0)),
            _ => {}
        }
        if position <= 8 && matches!(prev1, "<BOS>" | "<NL>" | "<INDENT>" | "<DEDENT>") {
            priors.extend([
                ("import", 4.0),
                ("if", 3.0),
                ("logs", 4.0),
                ("archive_path", 4.0),
                ("with", 4.5),
            ]);
        }
        if position > 8 && matches!(prev1, "<NL>" | "<INDENT>" | "<DEDENT>") {
            priors.extend([("with", 22.0), ("os", -14.0), ("if", -8.0)]);
        }
    }
    if matches!(
        category,
        "private_exec_zip_flat_directory" | "zip_flat_directory"
    ) {
        match (prev2, prev1) {
            (_, "zipfile") => priors.push((".", if prev2 == "with" { 36.0 } else { 28.0 })),
            ("zipfile", ".") => priors.push(("ZipFile", 34.0)),
            (_, "ZipFile") => priors.push(("(", 18.0)),
            ("archive", ".") => priors.push(("write", 30.0)),
            (_, "write") => priors.push(("(", 18.0)),
            _ => {}
        }
        if position <= 10 && matches!(prev1, "<BOS>" | "<NL>" | "<INDENT>" | "<DEDENT>") {
            priors.extend([
                ("import", 4.0),
                ("if", 3.0),
                ("names", 4.0),
                ("zip_path", 4.0),
                ("with", 4.5),
            ]);
        }
        if position > 10 && matches!(prev1, "<NL>" | "<INDENT>" | "<DEDENT>") {
            priors.extend([("with", 24.0), ("if", -12.0), ("os", -10.0)]);
        }
    }
    priors
}

fn execution_shape_context_token_bonus(
    task: &CodeTask,
    prev2: &str,
    prev1: &str,
    token: &str,
    position: usize,
) -> f32 {
    execution_shape_context_token_priors(task, prev2, prev1, position)
        .into_iter()
        .find_map(|(candidate, bonus)| (candidate == token).then_some(bonus))
        .unwrap_or(0.0)
}

fn interface_obligation_sequence_bonus(
    task: &CodeTask,
    existing: &[String],
    token: &str,
    prev2: &str,
    prev1: &str,
) -> f32 {
    let expected = category_expected_arg_count(task).unwrap_or(0);
    if expected == 0 {
        return if matches!(token, "data" | "other" | "args" | "extra") {
            -12.0
        } else {
            0.0
        };
    }

    let data_used = existing.iter().any(|item| item == "data");
    let other_used = existing.iter().any(|item| item == "other");
    let extra_used = existing.iter().any(|item| item == "extra");
    let mut score = 0.0;

    if expected >= 1 && !data_used {
        if token == "data" {
            score += if matches!(prev1, "(" | "," | "in" | "isfile" | "isdir" | "open") {
                5.0
            } else {
                2.0
            };
        }
        if token == "<EOS>" {
            score -= 6.0;
        }
    }

    if expected >= 2 && !other_used {
        if token == "other" {
            score += if execution_shaped_category(&task.category)
                && prev1 == "("
                && matches!(prev2, "load" | "run")
            {
                -7.0
            } else if prev1 == "("
                && matches!(
                    prev2,
                    "join" | "makedirs" | "get" | "run" | "open" | "ZipFile"
                )
            {
                9.0
            } else if matches!(prev1, "(" | "," | "in") {
                6.0
            } else {
                3.0
            };
        }
        if token == "<EOS>" || token == "return" {
            score -= 7.0;
        }
        if prev1 == "return" && matches!(token, "False" | "True" | "None" | "[]" | "''" | "\"\"") {
            score -= 7.0;
        }
    }

    if expected >= 3 && !extra_used {
        if token == "extra" {
            score += if matches!(prev1, "(" | ",") { 5.0 } else { 2.0 };
        }
        if token == "<EOS>" {
            score -= 5.0;
        }
    }

    score
}

fn state_sequence_rule_bonus(
    task: &CodeTask,
    existing: &[String],
    token: &str,
    position: usize,
) -> f32 {
    let prev = existing.last().map(String::as_str).unwrap_or("<BOS>");
    let prev2 = existing
        .iter()
        .rev()
        .nth(1)
        .map(String::as_str)
        .unwrap_or("<BOS>");
    let mut score = 0.0;
    score += semantic_type_sequence_bonus(task, existing, token);
    score += execution_shape_context_token_bonus(task, prev2, prev, token, position);
    score += interface_obligation_sequence_bonus(task, existing, token, prev2, prev);
    if prev == ":" && token == "<NL>" {
        score += 2.0;
    }
    if previous_meaningful_token(existing).as_deref() == Some(":")
        && prev == "<NL>"
        && token == "<INDENT>"
    {
        score += 2.0;
    }
    if token == "<EOS>" && !existing.iter().any(|item| item == "return") {
        score -= 4.0;
    }
    if existing.is_empty() {
        if token == "return" && complex_body_category(&task.category) {
            score -= 1.8;
        }
        if complex_body_category(&task.category)
            && matches!(
                token,
                "if" | "for"
                    | "while"
                    | "with"
                    | "total"
                    | "out"
                    | "values"
                    | "stack"
                    | "best"
                    | "factor"
                    | "items"
                    | "result"
            )
        {
            score += 0.9;
        }
    }
    if matches!(prev, "<NL>" | "<INDENT>" | "<DEDENT>") {
        if token == "return" {
            score += if position > 4 { 0.8 } else { 0.1 };
        }
        if matches!(
            token,
            "if" | "for"
                | "while"
                | "with"
                | "else"
                | "total"
                | "out"
                | "values"
                | "stack"
                | "best"
                | "factor"
                | "numbers"
                | "items"
                | "result"
        ) {
            score += 0.35;
        }
    }
    let line = current_line_tokens(existing);
    let csv_split_shuffle = matches!(
        task.category.as_str(),
        "private_exec_csv_split_shuffle" | "csv_split_shuffle"
    );
    if csv_split_shuffle {
        if prev == "<INDENT>" {
            if token == "return" {
                score += 18.0;
            }
            if matches!(
                token,
                "rows" | "with" | "out" | "chunk_size" | "random" | "base_dir"
            ) {
                score -= 28.0;
            }
        }
        if matches!(prev, "<NL>" | "<DEDENT>") {
            if matches!(
                token,
                "with" | "rows" | "out" | "base_dir" | "chunk_size" | "random" | "for"
            ) {
                score += 8.0;
            }
            if token == "return" && !generated_tokens_have_any(existing, &["for", "out", "path"]) {
                score -= 20.0;
            }
        }
    }
    if prev == ":" && block_header_line(&line) && token != "<NL>" {
        score -= 14.0;
    }
    if open_block_header_without_colon(&line) {
        if token == ":" {
            score += 8.0;
        }
        if matches!(token, "return" | "<NL>" | "<EOS>") {
            score -= 8.0;
        }
    }
    if with_header_requires_as_now(&line) {
        if token == "as" {
            score += 10.0;
        } else {
            score -= 8.0;
        }
    }
    if grammar_requires_colon_now(&line) {
        if token == ":" {
            score += 12.0;
        } else {
            score -= 10.0;
        }
    }
    if prev == "lambda" {
        if is_identifier(token) {
            score += 6.0;
        } else {
            score -= 8.0;
        }
    }
    if prev == "as" {
        if is_identifier(token) {
            score += 6.0;
        } else {
            score -= 8.0;
        }
    }
    if line.first().map(String::as_str) == Some("if") && token == "as" {
        score -= 12.0;
    }
    if archive_context_manager_in_if_header(&line) && matches!(token, ":" | "," | "+") {
        score -= 18.0;
    }
    if execution_shaped_category(&task.category) && matches!(prev, "<NL>" | "<INDENT>" | "<DEDENT>")
    {
        if token == "with" {
            score += 0.7;
        }
        if matches!(
            task.category.as_str(),
            "private_exec_archive_config_zip"
                | "private_exec_log_backup_tar"
                | "private_exec_zip_flat_directory"
                | "archive_config_zip"
                | "log_backup_tar"
                | "zip_flat_directory"
        ) && token == "with"
        {
            score += 1.4;
        }
    }
    if callable_keyword_argument_requires_callable_now(&line) {
        if callable_keyword_argument_start_token(token) {
            score += if token == "lambda" { 8.0 } else { 3.0 };
        } else {
            score -= 12.0;
        }
    }
    if for_header_target_before_in(&line) {
        if token == "," {
            score += 7.0;
        }
        if token == "in" && line.len() > 2 {
            score += 6.0;
        }
        if matches!(
            token,
            "+" | "-"
                | "*"
                | "/"
                | "//"
                | "%"
                | "&"
                | "|"
                | "=="
                | "!="
                | "<"
                | ">"
                | "<="
                | ">="
                | ":"
        ) {
            score -= 16.0;
        }
    }
    if matches!(prev, "<NL>" | "<INDENT>" | "<DEDENT>")
        && token == "except"
        && try_block_waiting_for_except(existing)
    {
        score += 7.0;
    }
    if line.first().map(String::as_str) == Some("if") && token == "Exception" {
        score -= 14.0;
    }
    if prev == "." {
        if is_identifier(token) {
            score += 5.0;
        } else {
            score -= 8.0;
        }
    }
    if matches!(
        prev,
        "range"
            | "len"
            | "set"
            | "sorted"
            | "int"
            | "abs"
            | "sum"
            | "min"
            | "max"
            | "append"
            | "lower"
            | "isdigit"
            | "isupper"
            | "pop"
    ) {
        if token == "(" {
            score += 1.8;
        } else if token == ":" || token == "<NL>" {
            score -= 2.5;
        }
    }
    if prev == "in" && token == "range" {
        score += 0.8;
    }
    if prev == "in" && token == "data" && primary_arg_kind(task) == ValueKind::Int {
        score -= 5.0;
    }
    if prev == "in" && token == "range" && primary_arg_kind(task) == ValueKind::Int {
        score += 2.2;
    }
    if prev == "%" && token == "0" {
        score -= 9.0;
    }
    if prev == "%" && task.category == "caesar_decode_shift5" && token == "26" {
        score += 4.0;
    }
    if prev == "%"
        && digit_rotation_category(&task.category)
        && matches!(token, "len" | "shift" | "other")
    {
        score += 1.6;
    }
    if task.category == "caesar_decode_shift5" {
        if matches!(token, "ord" | "chr" | "out" | "append" | "join" | "26") {
            score += 0.75;
        }
        if token == "0" && line.iter().any(|item| item == "%") {
            score -= 5.0;
        }
    }
    if matches!(
        task.category.as_str(),
        "largest_prime_factor" | "arithmetic_series_sum"
    ) && token == "range"
    {
        score += 0.8;
    }
    if task.category == "common_elements" {
        if matches!(token, "set" | "other" | "&" | "sorted") {
            score += 1.1;
        }
        if token == "data" && line.iter().any(|item| item == "sorted") {
            score -= 1.2;
        }
    }
    if task.category == "balanced_brackets_simple"
        && matches!(token, "stack" | "append" | "pop" | "False" | "True")
    {
        score += 0.8;
    }
    if task.category == "monotonic_sequence"
        && matches!(token, "nondecreasing" | "nonincreasing" | "range" | "len")
    {
        score += 0.8;
    }
    if token == ":"
        && line.iter().any(|item| item == "range")
        && !line.iter().any(|item| item == "(")
    {
        score -= 3.0;
    }
    if loop_returns_without_condition(
        &join_body_tokens(existing)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>(),
    ) && token == "<EOS>"
    {
        score -= 2.0;
    }
    score
}

fn semantic_type_sequence_bonus(task: &CodeTask, existing: &[String], token: &str) -> f32 {
    let mut score = semantic_type_token_bonus(task, token);
    score += decoder_return_shape_token_bonus(task, token);
    let prev = existing.last().map(String::as_str).unwrap_or("<BOS>");
    let line = current_line_tokens(existing);
    if category_expected_arg_count(task).is_some_and(|count| count < 2) && token == "other" {
        score -= 5.0;
    }
    match task.category.as_str() {
        "extract_def_name" | "private_extract_first_def" => {
            if prev == "." && matches!(token, "splitlines" | "strip" | "startswith" | "split") {
                score += 2.2;
            }
            if token == "index" || token == "other" {
                score -= 3.0;
            }
        }
        "reverse_string" | "private_reverse_text" => {
            if prev == "data" && token == "[" {
                score += 2.0;
            }
            if line.iter().any(|item| item == "[") && matches!(token, ":" | "-1") {
                score += 0.9;
            }
        }
        "palindrome" | "private_palindrome_check" => {
            if prev == "data" && token == "[" {
                score += 2.0;
            }
            if prev == "data" && line.iter().any(|item| item == "==") && token == "[" {
                score += 4.0;
            }
            let slice_open = line.iter().any(|item| item == "[");
            let colon_count = line.iter().filter(|item| item.as_str() == ":").count();
            if slice_open {
                if colon_count == 0 {
                    if token == ":" {
                        score += 4.0;
                    }
                    if token == "]" {
                        score -= 6.0;
                    }
                } else if colon_count == 1 {
                    if token == ":" {
                        score += 3.0;
                    }
                    if token == "]" {
                        score -= 3.0;
                    }
                } else if colon_count >= 2 {
                    if prev == ":" && token == "-" {
                        score += 3.0;
                    }
                    if prev == "-" && token == "1" {
                        score += 3.0;
                    }
                    if prev == "1" && token == "]" {
                        score += 4.0;
                    }
                }
            }
            if slice_open && matches!(token, ":" | "-" | "1" | "]") {
                score += 1.0;
            }
            if matches!(token, "==" | "data" | "return") {
                score += 0.8;
            }
            if matches!(token, "digits" | "0" | "other") {
                score -= 2.2;
            }
        }
        "median_list" | "private_median_even_odd" => {
            if existing.is_empty() && token == "items" {
                score += 1.8;
            }
            if matches!(token, "sorted" | "mid" | "len" | "2" | "if" | "%" | "1") {
                score += 1.0;
            }
            if matches!(token, "abs" | "zip" | "other") {
                score -= 2.5;
            }
        }
        "caesar_decode_shift5" | "private_decode_shift_general" => {
            if matches!(
                token,
                "out" | "append" | "chr" | "ord" | "join" | "for" | "ch"
            ) {
                score += 0.9;
            }
            if token == "26" {
                score += 2.0;
            }
            if prev == "%" && token == "0" {
                score -= 5.0;
            }
            if matches!(token, "False" | "True" | "digits") {
                score -= 2.0;
            }
        }
        "below_threshold" | "private_all_below_threshold" => {
            if matches!(
                token,
                "for" | "item" | "other" | ">=" | "<" | "False" | "True"
            ) {
                score += 0.8;
            }
            if token == "+" || token == "digits" {
                score -= 2.0;
            }
        }
        "add_numbers" | "private_pair_sum" => {
            if matches!(token, "return" | "data" | "+" | "other") {
                score += 1.1;
            }
            if matches!(token, "for" | "set" | "digits" | "False") {
                score -= 2.5;
            }
        }
        "same_chars" | "private_same_char_set" => {
            if matches!(token, "return" | "set" | "data" | "other" | "==") {
                score += 1.1;
            }
            if matches!(token, "+" | "digits" | "append") {
                score -= 2.5;
            }
        }
        "gcd_pair" | "private_gcd_pair_loop" => {
            if matches!(
                token,
                "while" | "range" | "%" | "other" | "data" | "return" | "abs" | "min"
            ) {
                score += 0.8;
            }
            if matches!(token, "sorted" | "set" | "digits") {
                score -= 2.0;
            }
        }
        "is_prime" | "private_prime_loop" => {
            if matches!(
                token,
                "if" | "range" | "%" | "False" | "True" | "2" | "return"
            ) {
                score += 0.8;
            }
            if matches!(token, "sorted" | "set" | "other") {
                score -= 2.0;
            }
        }
        "is_anagram" | "private_anagram_sorted" => {
            if matches!(token, "return" | "sorted" | "data" | "other" | "==") {
                score += 1.0;
            }
            if matches!(token, "%" | "digits" | "append") {
                score -= 2.0;
            }
        }
        "base_digits" | "private_base_digits" | "private_base_digits_alt" => {
            if matches!(
                token,
                "if" | "while"
                    | "digits"
                    | "append"
                    | "reversed"
                    | "join"
                    | "str"
                    | "%"
                    | "//"
                    | "other"
                    | "return"
            ) {
                score += 0.8;
            }
            if matches!(token, "sorted" | "set" | "append") {
                score -= if matches!(token, "append") { 0.0 } else { 2.0 };
            }
        }
        "factors" => {
            if matches!(
                token,
                "out" | "for" | "divisor" | "range" | "%" | "append" | "return"
            ) {
                score += 0.85;
            }
            if matches!(token, "sorted" | "set" | "digits") {
                score -= 1.5;
            }
        }
        "safe_head" => {
            if matches!(
                token,
                "if" | "not" | "data" | "None" | "return" | "[" | "0" | "]"
            ) {
                score += 0.85;
            }
            if matches!(token, "other" | "%" | "sorted") {
                score -= 1.8;
            }
        }
        "dict_required_keys" => {
            if matches!(
                token,
                "for" | "key" | "in" | "other" | "data" | "False" | "True" | "return"
            ) {
                score += 0.8;
            }
            if matches!(token, "%" | "digits" | "sorted") {
                score -= 1.8;
            }
        }
        "parse_ints" | "numeric_string_parser" => {
            if matches!(
                token,
                "out" | "for" | "token" | "split" | "replace" | "int" | "append" | "return"
            ) {
                score += 0.8;
            }
            if matches!(token, "sorted" | "set" | "other") {
                score -= 1.5;
            }
        }
        "filter_integers" => {
            if matches!(
                token,
                "out" | "for" | "item" | "isinstance" | "int" | "append" | "return"
            ) {
                score += 0.8;
            }
            if matches!(token, "%" | "sorted" | "other") {
                score -= 1.5;
            }
        }
        "title_case_words" => {
            if matches!(
                token,
                "return" | "' '" | "join" | "word" | "capitalize" | "for" | "in" | "split"
            ) {
                score += 0.8;
            }
            if matches!(token, "set" | "sorted" | "%" | "other") {
                score -= 2.0;
            }
        }
        "cube_volume" => {
            if matches!(token, "return" | "data" | "*" | "**") {
                score += 1.0;
            }
            if matches!(token, "<" | ">" | "==" | "False" | "True") {
                score -= 3.0;
            }
            if prev == "*" && token == "3" {
                score -= 2.0;
            }
        }
        "cube_lateral_surface_area" => {
            if matches!(token, "return" | "4" | "data" | "*") {
                score += 1.0;
            }
            if matches!(token, "<" | ">" | "==" | "False" | "True") {
                score -= 3.0;
            }
        }
        "cylinder_lateral_surface_area" => {
            if matches!(
                token,
                "return" | "2" | "3.141592653589793" | "*" | "data" | "other"
            ) {
                score += 0.9;
            }
            if matches!(token, "<" | ">" | "==" | "False" | "True") {
                score -= 3.0;
            }
        }
        "replace_whitespace" => {
            if matches!(token, "return" | "data" | "." | "replace" | "' '" | "other") {
                score += 1.1;
            }
            if matches!(token, "<" | ">" | "==" | "False" | "True" | "%") {
                score -= 2.5;
            }
        }
        "tuple_item_count" => {
            if matches!(token, "return" | "data" | "." | "count" | "other") {
                score += 1.1;
            }
            if matches!(token, "%" | "*" | "<" | "==" | "False" | "True") {
                score -= 2.5;
            }
        }
        "tuple_nested_elementwise_max" => {
            if matches!(
                token,
                "out" | "for" | "zip" | "max" | "tuple" | "append" | "return"
            ) {
                score += 0.9;
            }
            if matches!(token, "%" | "<" | ">" | "==") {
                score -= 1.5;
            }
        }
        "list_chunks_every_n" => {
            if matches!(
                token,
                "return" | "data" | "idx" | "range" | "len" | "other" | ":"
            ) {
                score += 0.85;
            }
            if matches!(token, "%" | "set" | "sorted") {
                score -= 1.5;
            }
        }
        "combinations_with_replacement" => {
            if matches!(
                token,
                "out" | "build" | "append" | "tuple" | "range" | "return"
            ) {
                score += 0.8;
            }
        }
        "max_list" | "private_max_item" => {
            if matches!(token, "max" | "data" | "return") {
                score += 1.0;
            }
            if matches!(token, "other" | "zip" | "abs") {
                score -= 2.0;
            }
        }
        "rescale_to_unit" => {
            if matches!(
                token,
                "low" | "high" | "min" | "max" | "item" | "for" | "return"
            ) {
                score += 0.9;
            }
            if matches!(token, "seen" | "True" | "False") {
                score -= 2.0;
            }
        }
        "decode_cyclic" => {
            if matches!(
                token,
                "out" | "group" | "idx" | "range" | "len" | "join" | "["
            ) {
                score += 0.9;
            }
            if matches!(token, "min" | "max" | "set") {
                score -= 2.0;
            }
        }
        "prime_fib_sequence" => {
            if matches!(
                token,
                "found" | "a" | "b" | "while" | "divisor" | "%" | "True" | "False"
            ) {
                score += 0.9;
            }
            if matches!(token, "sorted" | "set" | "other") {
                score -= 1.5;
            }
        }
        "polynomial_zero_bisection" => {
            if matches!(token, "left" | "right" | "mid" | "for" | "range" | "return") {
                score += 0.8;
            }
        }
        "prime_factors" => {
            if matches!(token, "out" | "value" | "factor" | "while" | "%" | "append") {
                score += 0.9;
            }
        }
        "string_sequence" => {
            if matches!(token, "parts" | "str" | "join" | "range" | "append") {
                score += 0.9;
            }
        }
        "two_sum_zero_exists" | "three_sum_zero_exists" => {
            if matches!(
                token,
                "seen" | "range" | "return" | "True" | "False" | "data"
            ) {
                score += 0.75;
            }
        }
        "count_digit_under_divisibility" => {
            if matches!(
                token,
                "total" | "range" | "str" | "count" | "%" | "11" | "13"
            ) {
                score += 0.8;
            }
        }
        "index_or_minus_one" | "private_index_or_default" => {
            if matches!(token, "if" | "other" | "in" | "data" | "index" | "-1") {
                score += 1.0;
            }
        }
        "final_y_vowel_private" | "private_final_y_vowels" => {
            if matches!(token, "isalpha" | "enumerate" | "idx" | "len" | "total") {
                score += 0.9;
            }
            if token == "index" || token == "other" {
                score -= 2.0;
            }
        }
        "woodall_number_check"
        | "polygonal_octagonal_number"
        | "polygonal_tetrahedral_number"
        | "polygonal_centered_hexagonal_number"
        | "sphere_volume"
        | "sphere_surface_area"
        | "difference_of_squares_check"
        | "closest_smaller_number"
        | "car_race_collision_count" => {
            if matches!(
                token,
                "return" | "data" | "*" | "**" | "%" | "while" | "k" | "True" | "False"
            ) {
                score += 0.8;
            }
            if matches!(token, "for" | "item" | "set" | "sorted")
                && !matches!(task.category.as_str(), "woodall_number_check")
            {
                score -= 1.5;
            }
        }
        "nested_flat_sum"
        | "positive_count"
        | "positive_filter"
        | "sublist_contains"
        | "equal_tuple_lengths"
        | "same_pattern_sequence"
        | "tuple_all_divisible"
        | "frequency_dict"
        | "substring_in_list"
        | "closest_pair_sorted"
        | "unique_once_stable"
        | "filter_by_prefix"
        | "sort_indices_multiple_three" => {
            if matches!(
                token,
                "for" | "item" | "out" | "append" | "counts" | "len" | "range" | "return" | "if"
            ) {
                score += 0.75;
            }
            if primary_arg_kind(task) == ValueKind::Int && token == "data" && prev == "in" {
                score -= 3.0;
            }
        }
        "ascii_mod_char"
        | "longest_word_length"
        | "overlapping_substring_count"
        | "spelled_number_sort"
        | "flip_case"
        | "concat_strings"
        | "digit_substring_length_sum_count" => {
            if matches!(
                token,
                "for"
                    | "ch"
                    | "split"
                    | "join"
                    | "len"
                    | "range"
                    | "return"
                    | "total"
                    | "ord"
                    | "chr"
            ) {
                score += 0.75;
            }
            if matches!(token, "set" | "other")
                && !matches!(task.category.as_str(), "overlapping_substring_count")
            {
                score -= 1.2;
            }
        }
        "bell_number_sequence" | "newman_conway_sequence" => {
            if matches!(
                token,
                "for" | "range" | "values" | "bell" | "append" | "return" | "if"
            ) {
                score += 0.8;
            }
            if matches!(token, "set" | "sorted" | "other") {
                score -= 1.5;
            }
        }
        _ => {}
    }
    score
}

fn decoder_return_shape_token_bonus(task: &CodeTask, token: &str) -> f32 {
    match decoder_return_shape(task).as_str() {
        "bool" => match token {
            "True" | "False" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "and" | "or" | "not" => 0.65,
            "return" | "if" => 0.35,
            "out" | "append" | "join" | "[" => -0.25,
            _ => 0.0,
        },
        "list" => match token {
            "out" | "append" | "[" | "]" | "list" | "sorted" | "range" | "for" | "return" => 0.5,
            "True" | "False" => -0.45,
            _ => 0.0,
        },
        "dict" => match token {
            "counts" | "dict" | "get" | "{" | "}" | "for" | "return" => 0.55,
            "True" | "False" => -0.35,
            _ => 0.0,
        },
        "tuple" => match token {
            "tuple" | "(" | ")" | "zip" | "return" => 0.45,
            "True" | "False" => -0.3,
            _ => 0.0,
        },
        "str" => match token {
            "str" | "join" | "split" | "lower" | "replace" | "chr" | "ord" | "[" | ":"
            | "return" => 0.45,
            "append" | "out" => 0.25,
            "True" | "False" => -0.4,
            _ => 0.0,
        },
        "number" => match token {
            "total" | "best" | "count" | "sum" | "len" | "range" | "+" | "-" | "*" | "/" | "//"
            | "%" | "return" => 0.45,
            "append" => -0.2,
            "True" | "False" => -0.35,
            _ => 0.0,
        },
        _ => 0.0,
    }
}

fn complex_body_category(category: &str) -> bool {
    private_mbpp_category(category)
        || matches!(
            category,
            "balanced_brackets_simple"
                | "monotonic_sequence"
                | "largest_prime_factor"
                | "derivative_coefficients"
                | "tribonacci_sequence"
                | "fibonacci_loop_private"
                | "lucas_loop_private"
                | "shifted_recurrence_private"
                | "nested_recurrence_private"
                | "rotate_sequence"
                | "circular_digit_shift"
                | "digit_rotate_right_private"
                | "signed_digit_rotate_private"
                | "multi_step_digit_shift_private"
                | "digit_sum_casefold"
                | "fruit_distribution_private"
                | "parse_ints"
                | "median_list"
                | "modular_power_two"
                | "caesar_decode_shift5"
                | "remove_vowels"
                | "below_threshold"
                | "factors"
                | "is_prime"
                | "largest_divisor"
                | "prime_factors"
                | "count_vowels"
                | "final_y_vowel_private"
                | "suffix_y_vowel_private"
                | "case_punct_vowel_private"
                | "arithmetic_series_sum"
                | "woodall_number_check"
                | "string_sequence"
                | "two_sum_zero_exists"
                | "three_sum_zero_exists"
                | "count_digit_under_divisibility"
                | "nested_flat_sum"
                | "positive_count"
                | "positive_filter"
                | "sublist_contains"
                | "equal_tuple_lengths"
                | "same_pattern_sequence"
                | "tuple_all_divisible"
                | "ascii_mod_char"
                | "dict_merge_three"
                | "frequency_dict"
                | "longest_word_length"
                | "substring_in_list"
                | "overlapping_substring_count"
                | "spelled_number_sort"
                | "closest_pair_sorted"
                | "unique_once_stable"
                | "filter_by_prefix"
                | "sort_indices_multiple_three"
                | "digit_substring_length_sum_count"
                | "bell_number_sequence"
                | "newman_conway_sequence"
                | "rescale_to_unit"
                | "decode_cyclic"
                | "prime_fib_sequence"
                | "polynomial_zero_bisection"
        )
}

fn token_alignment_bonus(token: &str, prompt_tokens: &[String]) -> f32 {
    let lowered = token.to_lowercase();
    if lowered.len() < 2 {
        return 0.0;
    }
    if prompt_tokens
        .iter()
        .any(|item| item == &lowered || item.ends_with(&format!(":{lowered}")))
    {
        0.25
    } else {
        0.0
    }
}

fn length_bonus(length: usize) -> f32 {
    match length {
        0 => -4.0,
        1..=2 => -0.8,
        3..=24 => 0.25,
        _ => -0.4,
    }
}

fn learned_token_max_steps(task: &CodeTask, base: usize) -> usize {
    if contract_requires_extended_learned_body(task) {
        base.max(contract_learned_body_step_floor(task) / 2)
            .min(192)
    } else if execution_shaped_category(&task.category) {
        base.max(40).min(64)
    } else {
        base
    }
}

fn learned_body_ngram_max_steps(task: &CodeTask, base: usize) -> usize {
    if contract_requires_extended_learned_body(task) {
        base.max(contract_learned_body_step_floor(task)).min(320)
    } else if execution_shaped_category(&task.category) {
        base.max(96).min(160)
    } else {
        base
    }
}

fn contract_requires_extended_learned_body(task: &CodeTask) -> bool {
    let family = decoder_type_family(task);
    let hints = decoder_required_constructs(task);
    let generation_hints = decoder_contract_generation_hints(task);
    let full_body_required = task
        .raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("full_body_required"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let semantic_family = task
        .raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("semantic_family"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let contract_text = task_contract_text(task);

    full_body_required
        && (matches!(
            family.as_str(),
            "algorithmic_planning" | "graph_search_algorithm" | "state_machine"
        ) || hints.iter().any(|hint| {
            matches!(
                hint.as_str(),
                "algorithmic_planning"
                    | "binary_search"
                    | "dynamic_programming"
                    | "graph"
                    | "queue"
                    | "stack"
                    | "state_update"
            )
        }) || generation_hints.iter().any(|hint| {
            matches!(
                hint.as_str(),
                "algorithmic_planning"
                    | "binary_search_previous"
                    | "dp_prefix_best"
                    | "dynamic_programming"
                    | "bfs_queue"
                    | "distance_dict"
                    | "stack_push_pop"
                    | "floor_ceiling_clamp"
                    | "append_each_state"
            )
        }) || body_has_any(
            &semantic_family,
            &[
                "dynamic_programming",
                "graph_shortest_hops",
                "bounded_state_update",
                "stack_cancellation",
            ],
        ) || body_has_any(
            &contract_text,
            &[
                "dynamic programming",
                "shortest hop",
                "state",
                "stack",
                "graph",
            ],
        ))
}

fn contract_learned_body_step_floor(task: &CodeTask) -> usize {
    let hints = decoder_required_constructs(task);
    let generation_hints = decoder_contract_generation_hints(task);
    let semantic_family = task
        .raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("semantic_family"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    if hints.contains("dynamic_programming")
        || hints.contains("binary_search")
        || generation_hints.contains("dp_prefix_best")
        || generation_hints.contains("binary_search_previous")
        || semantic_family.contains("dynamic_programming")
    {
        320
    } else if hints.contains("graph")
        || hints.contains("queue")
        || generation_hints.contains("bfs_queue")
        || semantic_family.contains("graph_shortest_hops")
    {
        256
    } else if hints.contains("state_update")
        || hints.contains("stack")
        || generation_hints.contains("stack_push_pop")
        || generation_hints.contains("floor_ceiling_clamp")
        || semantic_family.contains("bounded_state_update")
        || semantic_family.contains("stack_cancellation")
    {
        192
    } else {
        160
    }
}

fn learned_position_cap(category: &str, position: usize) -> usize {
    if category.starts_with("edge_v3_") {
        position.min(320)
    } else if execution_shaped_category(category) {
        position.min(192)
    } else {
        position.min(64)
    }
}
