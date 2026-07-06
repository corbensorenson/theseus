fn compact_visible_arg_pairs(task: &CodeTask) -> Vec<(String, String)> {
    let mut pairs = vec![("data".to_string(), "other".to_string())];
    let entry = sanitize_ident(&task.entry_point);
    let args = visible_signature(&entry, &task.prompt).arg_names;
    let start = if args.first().is_some_and(|arg| arg == "self") && args.len() >= 3 {
        1
    } else {
        0
    };
    if args.len().saturating_sub(start) >= 2 {
        pairs.push((args[start].to_lowercase(), args[start + 1].to_lowercase()));
    }
    pairs.sort();
    pairs.dedup();
    pairs
}

fn compact_add_pair_expr_ok(task: &CodeTask, expr: &str, body_compact: &str) -> bool {
    compact_visible_arg_pairs(task)
        .into_iter()
        .any(|(left, right)| {
            let direct = format!("{left}+{right}");
            let reverse = format!("{right}+{left}");
            let total_direct = format!("total={left}+{right}");
            let total_reverse = format!("total={right}+{left}");
            expr.contains(&direct)
                || expr.contains(&reverse)
                || (expr == "total"
                    && (body_compact.contains(&total_direct)
                        || body_compact.contains(&total_reverse)))
        })
}

fn compact_same_chars_pair_expr_ok(task: &CodeTask, expr: &str) -> bool {
    compact_visible_arg_pairs(task)
        .into_iter()
        .any(|(left, right)| {
            expr.contains(&format!("set({left})"))
                && expr.contains(&format!("set({right})"))
                && expr.contains("==")
        })
}

fn bogus_return_attribute_body(body: &str) -> bool {
    let Some(expr) = first_top_level_return_expr(body) else {
        return false;
    };
    bogus_return_attribute_expr(&expr)
}

fn bogus_return_local_callable_body(body: &str) -> bool {
    let Some(expr) = first_top_level_return_expr(body) else {
        return false;
    };
    let compact = expr
        .to_lowercase()
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let Some(open_idx) = compact.find('(') else {
        return false;
    };
    let callee = &compact[..open_idx];
    if callee.is_empty()
        || !is_identifier(callee)
        || matches!(
            callee,
            "abs"
                | "all"
                | "any"
                | "bool"
                | "dict"
                | "enumerate"
                | "filter"
                | "float"
                | "int"
                | "len"
                | "list"
                | "map"
                | "max"
                | "min"
                | "pow"
                | "range"
                | "reversed"
                | "round"
                | "set"
                | "sorted"
                | "str"
                | "sum"
                | "tuple"
                | "zip"
        )
    {
        return false;
    }
    assigned_names_from_body(body)
        .into_iter()
        .any(|name| name.eq_ignore_ascii_case(callee))
        || body_name_assigned(&body.to_lowercase(), callee)
}

fn bogus_return_attribute_expr(expr: &str) -> bool {
    let compact = expr
        .to_lowercase()
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    if compact.contains('(') || compact.contains('[') || compact.contains('{') {
        return false;
    }
    let Some((base, attr)) = compact.rsplit_once('.') else {
        return false;
    };
    if base.is_empty()
        || attr.is_empty()
        || base
            .chars()
            .any(|ch| !ch.is_ascii_alphanumeric() && ch != '_')
    {
        return false;
    }
    matches!(
        attr,
        "isinstance"
            | "list"
            | "dict"
            | "tuple"
            | "str"
            | "int"
            | "float"
            | "bool"
            | "set"
            | "len"
            | "sum"
            | "min"
            | "max"
            | "sorted"
            | "range"
            | "append"
            | "extend"
            | "insert"
            | "remove"
            | "pop"
            | "sort"
            | "reverse"
            | "items"
            | "keys"
            | "values"
            | "get"
            | "split"
            | "strip"
            | "lower"
            | "upper"
            | "replace"
            | "join"
    )
}

fn repeated_self_multiplication_expr(expr: &str) -> bool {
    let compact = expr
        .to_lowercase()
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    compact.matches("data*data").count() >= 1
        || compact.matches("text*text").count() >= 1
        || compact.matches("items*items").count() >= 1
}

fn balanced_bracket_stack_discipline_ok(body: &str) -> bool {
    let mut saw_loop = false;
    let mut saw_nested_pop = false;
    let mut previous_nonempty: Option<(usize, String)> = None;
    for raw_line in body.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let indent = raw_line.chars().take_while(|ch| *ch == ' ').count() / 4;
        if line.starts_with("for ") && line.ends_with(':') {
            saw_loop = true;
        }
        if line.starts_with("stack.pop") {
            let follows_mismatch_guard =
                previous_nonempty
                    .as_ref()
                    .is_some_and(|(prev_indent, prev_line)| {
                        *prev_indent > indent && prev_line.starts_with("return False")
                    });
            if !follows_mismatch_guard {
                return false;
            }
            saw_nested_pop = true;
        }
        previous_nonempty = Some((indent, line.to_string()));
    }
    saw_loop && saw_nested_pop
}

fn scalar_loop_over_data(body: &str) -> bool {
    body.lines().any(|line| {
        let compact = line.trim().split_whitespace().collect::<Vec<_>>().join(" ");
        compact.starts_with("for ") && compact.ends_with(" in data:")
    })
}

fn body_name_assigned(body: &str, name: &str) -> bool {
    body.lines().any(|line| {
        let trimmed = line.trim_start();
        trimmed.starts_with(&format!("{name} ="))
            || trimmed.starts_with(&format!("{name} +="))
            || trimmed.starts_with(&format!("{name} -="))
            || trimmed.starts_with(&format!("{name} *="))
            || trimmed.starts_with(&format!("{name} /="))
    })
}

fn uses_prompt_visible_argument_names(task: &CodeTask, body: &str) -> bool {
    let entry = sanitize_ident(&task.entry_point);
    let signature = visible_signature(&entry, &task.prompt);
    signature
        .arg_names
        .iter()
        .any(|name| !matches!(name.as_str(), "data" | "other" | "args") && body.contains(name))
}
