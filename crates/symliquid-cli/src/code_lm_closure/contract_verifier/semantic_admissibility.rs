use super::*;

pub(super) fn body_semantically_admissible(task: &CodeTask, body: &str) -> bool {
    let lowered = body.to_lowercase();
    let compact = lowered
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    if edge_v3_family_has_strict_contract(task) {
        return edge_v3_family_body_semantically_admissible(task, &lowered, &compact);
    }
    let scalar_input_family = matches!(
        category_semantic_family(&task.category),
        "scalar_numeric" | "scalar_recurrence"
    );
    if compact.contains("%0") || compact.contains("%(0") {
        return false;
    }
    if compact.contains("nondecreasingornonincreasing=")
        || compact.contains("nondecreasingandnonincreasing=")
    {
        return false;
    }
    if category_expected_arg_count(task).is_some_and(|count| count < 2)
        && body_mentions_token(body, "other")
    {
        return false;
    }
    if lowered.contains("return data.append")
        || lowered.contains("return out.append")
        || lowered.contains("return values.append")
        || lowered.contains("return stack.append")
    {
        return false;
    }
    if primary_arg_kind(task) == ValueKind::Int || scalar_input_family {
        if scalar_loop_over_data(body) || lowered.contains("len(data)") {
            return false;
        }
        if lowered.contains("data.split") || lowered.contains("sorted(data)") {
            return false;
        }
        if compact.contains("data[::-1]")
            || compact.contains(".index(")
            || compact.contains("indata")
            || lowered.contains(" in data")
            || lowered.contains(" in  data")
        {
            return false;
        }
    }
    if primary_arg_kind(task) == ValueKind::List && lowered.contains("data.split") {
        return false;
    }
    if matches!(primary_arg_kind(task), ValueKind::List | ValueKind::Str)
        || matches!(
            category_semantic_family(&task.category),
            "collection_transform" | "string_transform" | "membership_lookup"
        )
    {
        if let Some(return_expr) = first_top_level_return_expr(body) {
            if repeated_self_multiplication_expr(&return_expr) {
                return false;
            }
        }
    }
    if primary_arg_kind(task) == ValueKind::Str
        && (lowered.contains("data <= 1") || lowered.contains("data < 2"))
    {
        return false;
    }
    if !digit_rotation_category(&task.category)
        && lowered.contains("digits = str")
        && lowered.matches("return digits").count() >= 2
    {
        return false;
    }
    let palindrome_like_category = task.category.contains("palindrome")
        || matches!(
            task.category.as_str(),
            "reverse_string" | "private_reverse_text"
        );
    if !palindrome_like_category
        && (compact.contains("data==data[::-1]")
            || compact.contains("data==data==data[::-1]")
            || compact.contains("text==text[::-1]"))
    {
        return false;
    }
    if matches!(
        task.category.as_str(),
        "remove_vowels" | "caesar_decode_shift5"
    ) && lowered.contains("return total")
    {
        return false;
    }
    if task.category == "below_threshold" && lowered.contains("return digits") {
        return false;
    }
    if recurrence_category(&task.category)
        && (lowered.contains("sorted(") || lowered.contains("sorted ("))
    {
        return false;
    }
    if task.category == "add_numbers" {
        if let Some(return_expr) = first_top_level_return_expr(body) {
            let expr = return_expr.replace(' ', "");
            if !compact_add_pair_expr_ok(task, &expr, &compact) {
                return false;
            }
        }
    }
    if task.category == "same_chars" {
        if let Some(return_expr) = first_top_level_return_expr(body) {
            let expr = return_expr.replace(' ', "").to_lowercase();
            if !compact_same_chars_pair_expr_ok(task, &expr) {
                return false;
            }
        }
    }
    if task.category == "replace_whitespace" {
        if let Some(return_expr) = first_top_level_return_expr(body) {
            let expr = return_expr.replace(' ', "").to_lowercase();
            if expr == "data" || !(expr.contains("replace(") || lowered.contains("for ")) {
                return false;
            }
        }
    }
    if task.category == "cube_volume" {
        if let Some(return_expr) = first_top_level_return_expr(body) {
            let expr = return_expr.replace(' ', "").to_lowercase();
            if expr.contains("<") || expr.contains(">") || expr.contains("==") || expr == "data*3" {
                return false;
            }
            let data_factor_count = expr.matches("data").count();
            if !(expr.contains("**3") || data_factor_count >= 3) {
                return false;
            }
        }
    }
    if task.category == "cube_lateral_surface_area" {
        if let Some(return_expr) = first_top_level_return_expr(body) {
            let expr = return_expr.replace(' ', "").to_lowercase();
            if expr.contains("<") || expr.contains(">") || expr.contains("==") {
                return false;
            }
            if !(expr.contains('4') && expr.matches("data").count() >= 2) {
                return false;
            }
        }
    }
    if task.category == "cylinder_lateral_surface_area" {
        if let Some(return_expr) = first_top_level_return_expr(body) {
            let expr = return_expr.replace(' ', "").to_lowercase();
            if expr.contains("<") || expr.contains(">") || expr.contains("==") {
                return false;
            }
            if !(expr.contains("data")
                && expr.contains("other")
                && (expr.contains("3.141") || expr.contains("pi")))
            {
                return false;
            }
        }
    }
    if task.category == "tuple_item_count" {
        if let Some(return_expr) = first_top_level_return_expr(body) {
            let expr = return_expr.replace(' ', "").to_lowercase();
            if expr.contains('%') || expr.contains("count(") && !expr.contains(".count(") {
                return false;
            }
            if !(expr.contains(".count(") || lowered.contains("for ")) {
                return false;
            }
        }
    }
    if task.category == "count_primes_below"
        && (compact.contains("**0)") || compact.contains("**0+") || compact.contains("**0)+"))
    {
        return false;
    }
    if task.category == "tuple_nested_elementwise_max" {
        if !(lowered.contains("zip(") && lowered.contains("max(") && lowered.contains("tuple")) {
            return false;
        }
    }
    if task.category == "palindrome" {
        if let Some(return_expr) = first_top_level_return_expr(body) {
            let expr = return_expr.replace(' ', "").to_lowercase();
            if expr == "data==data" || expr == "text==text" || expr.contains("data[]") {
                return false;
            }
        }
    }
    if matches!(task.category.as_str(), "median_list" | "median_odd") {
        if let Some(return_expr) = first_top_level_return_expr(body) {
            let expr = return_expr.replace(' ', "").to_lowercase();
            if expr == "items" || expr == "data" || expr.starts_with("sorted(") {
                return false;
            }
        }
        if body.lines().any(|line| line.trim() == "return items") {
            return false;
        }
        if compact.contains("abs(data)")
            || compact.contains("data%")
            || lowered.contains("zip(data")
        {
            return false;
        }
    }
    if matches!(task.category.as_str(), "max_list" | "min_list" | "sum_list")
        && (compact.contains("abs(data)")
            || compact.contains("data%")
            || lowered.contains("zip(data"))
    {
        return false;
    }
    if matches!(
        task.category.as_str(),
        "reverse_string" | "palindrome" | "extract_def_name"
    ) && (compact.contains("abs(data)") || lowered.contains("zip(data"))
    {
        return false;
    }
    if task.category == "extract_def_name"
        && (lowered.contains("data.index") || lowered.contains(" in data else -1"))
    {
        return false;
    }
    if task.category == "below_threshold"
        && (lowered.contains("data + other") || lowered.contains("return data"))
    {
        return false;
    }
    if task.category == "dict_merge_three" && !lowered.contains("update(") {
        return false;
    }
    if task.category == "top_k_largest"
        && !(compact.contains("[:other]")
            || compact.contains("[:limit]")
            || compact.contains("[:k]")
            || compact.contains("[:n]"))
    {
        return false;
    }
    if task.category == "list_chunks_every_n"
        && !(lowered.contains("range(0")
            && (lowered.contains("idx:idx") || lowered.contains("idx: idx")))
    {
        return false;
    }
    if task.category == "safe_head"
        && !(lowered.contains("if ")
            && (lowered.contains("return fallback") || lowered.contains("return other")))
    {
        return false;
    }
    if task.category == "stable_dedupe"
        && !(lowered.contains("seen") && lowered.contains("not in seen"))
    {
        return false;
    }
    if task.category == "flatten_once" && !lowered.contains("extend(") {
        return false;
    }
    if task.category == "title_case_words"
        && !(lowered.contains(".title(") || lowered.contains(".capitalize("))
    {
        return false;
    }
    if task.category == "count_truthy"
        && (compact.contains("returncounts") || lowered.contains("counts = {}"))
    {
        return false;
    }
    if task.category == "frequency_at_least_value"
        && !(lowered.contains("counts") && lowered.contains("best") && lowered.contains("count >="))
    {
        return false;
    }
    if task.category == "rescale_to_unit" {
        if !lowered.contains("min(") && !lowered.contains("low") {
            return false;
        }
        if lowered.contains("seen = set") || lowered.contains("return true") {
            return false;
        }
    }
    if task.category == "decode_cyclic"
        && (compact.contains("returnmin(data)") || compact.contains("returnmax(data)"))
    {
        return false;
    }
    if task.category == "largest_divisor" && lowered.contains("str(") {
        return false;
    }
    if task.category == "remove_vowels" && lowered.contains("aeiou") && !lowered.contains("lower") {
        return false;
    }
    if task.category == "common_elements" {
        if compact.contains("returnsorted(data)") || compact.contains("return(data)") {
            return false;
        }
        if lowered.contains("sorted") && !lowered.contains("other") {
            return false;
        }
    }
    if task.category == "all_prefixes"
        && !(compact.contains("[:")
            && (compact.contains("range(") || compact.contains("enumerate(")))
        && !(lowered.contains("current")
            && (compact.contains("current+=") || compact.contains("current=current+"))
            && compact.contains("append(current)"))
    {
        return false;
    }
    if let Some(return_expr) = first_top_level_return_expr(body) {
        let normalized_return = return_expr.trim().to_ascii_lowercase();
        if is_identifier(&return_expr)
            && !matches!(normalized_return.as_str(), "true" | "false" | "none")
            && !body_name_assigned(body, &return_expr)
            && !visible_signature_arg_names(task).contains(&return_expr)
            && !matches!(return_expr.as_str(), "data" | "other" | "args" | "extra")
        {
            return false;
        }
        match task.category.as_str() {
            "arithmetic_series_sum" | "sum_list" if body_name_assigned(body, "total") => {
                if return_expr != "total" {
                    return false;
                }
            }
            "derivative_coefficients" if body_name_assigned(body, "out") => {
                if return_expr != "out" {
                    return false;
                }
            }
            "largest_prime_factor" if body_name_assigned(body, "best") => {
                if return_expr != "best" {
                    return false;
                }
            }
            _ => {}
        }
    }
    if task.category == "balanced_brackets_simple" && !balanced_bracket_stack_discipline_ok(body) {
        return false;
    }
    if matches!(
        task.category.as_str(),
        "largest_prime_factor" | "arithmetic_series_sum" | "is_prime" | "factors"
    ) && scalar_loop_over_data(body)
    {
        return false;
    }
    if compact.contains("return(data)") && !simple_return_category(&task.category) {
        return false;
    }
    if compact.contains("returnbest") && !body_name_assigned(body, "best") {
        return false;
    }
    if compact.contains("returntotal") && !body_name_assigned(body, "total") {
        return false;
    }
    if compact.contains("returnnondecreasing") && !body_name_assigned(body, "nondecreasing") {
        return false;
    }
    if compact.contains("returnnonincreasing") && !body_name_assigned(body, "nonincreasing") {
        return false;
    }
    true
}

fn edge_v3_family_has_strict_contract(task: &CodeTask) -> bool {
    matches!(
        task.category.as_str(),
        "edge_v3_weighted_interval_best"
            | "edge_v3_graph_distance_labels"
            | "edge_v3_capped_running_balance"
            | "edge_v3_stack_cancel_tokens"
    )
}

fn edge_v3_family_body_semantically_admissible(
    task: &CodeTask,
    lowered: &str,
    compact: &str,
) -> bool {
    match task.category.as_str() {
        "edge_v3_weighted_interval_best" => {
            if compact.contains("len(item)<=start")
                || compact.contains("returnbest")
                || compact.contains("best=dpreturn")
            {
                return false;
            }
            body_has_any(lowered, &["jobs.sort", "jobs.sort ("])
                && body_has_any(lowered, &["dp.append", "dp.append ("])
                && body_has_any(lowered, &["ends.append", "ends.append ("])
                && body_has_any(lowered, &["while lo < hi", "while lo <hi"])
                && compact.contains("iflen(item)>=3")
                && compact.contains("best=dp[lo]+weight")
                && compact.contains("returndp[-1]")
        }
        "edge_v3_graph_distance_labels" => {
            body_has_any(lowered, &["deque", "queue"])
                && body_has_any(lowered, &["graph.setdefault", "graph . setdefault"])
                && body_has_any(lowered, &["graph.get", "graph . get"])
                && body_has_any(lowered, &["popleft", ".pop(0)", ".pop (0)"])
                && body_has_any(lowered, &["queue.append", "queue . append"])
                && compact.contains("dist={start:0}")
                && compact.contains("dist[nxt]=dist[node]+1")
                && compact.contains("returndist")
        }
        "edge_v3_capped_running_balance" => {
            body_has_any(
                lowered,
                &["balance += delta", "balance +=delta", "balance+=delta"],
            ) && body_has_any(lowered, &["out.append", "out . append"])
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
                && body_has_any(lowered, &["stack.pop", "stack . pop"])
                && body_has_any(lowered, &["stack.append", "stack . append"])
                && compact.contains("returnstack")
        }
        _ => true,
    }
}

pub(super) fn visible_signature_arg_names(task: &CodeTask) -> BTreeSet<String> {
    let entry = sanitize_ident(&task.entry_point);
    visible_signature(&entry, &task.prompt)
        .arg_names
        .into_iter()
        .collect()
}

pub(super) fn first_top_level_return_expr(body: &str) -> Option<String> {
    for raw_line in body.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let indent = raw_line.chars().take_while(|ch| *ch == ' ').count() / 4;
        if indent == 0 {
            if let Some(expr) = line.strip_prefix("return ") {
                return Some(expr.trim().to_string());
            }
        }
    }
    None
}
