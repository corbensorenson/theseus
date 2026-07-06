// Visible Python signatures/imports used to keep generated candidates aligned with the prompt contract.

use super::*;

pub(in crate::code_lm_closure) fn visible_prompt_import_prelude(prompt: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for raw_line in prompt.lines() {
        let line = raw_line.trim();
        if line.starts_with("def ") || line.starts_with("class ") {
            break;
        }
        if let Some(clean) = safe_visible_import_line(line) {
            if seen.insert(clean.clone()) {
                out.push(clean);
            }
        }
    }
    out
}

pub(in crate::code_lm_closure) fn safe_visible_import_line(line: &str) -> Option<String> {
    if line.starts_with("import ") {
        let mut parts = Vec::new();
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
            if safe_visible_import_module(base) {
                parts.push(item.to_string());
            }
        }
        if parts.is_empty() {
            return None;
        }
        return Some(format!("import {}", parts.join(", ")));
    }
    if line.starts_with("from ") {
        let Some((module_part, names_part)) =
            line.trim_start_matches("from ").split_once(" import ")
        else {
            return None;
        };
        let module = module_part.trim();
        let base = module.split('.').next().unwrap_or("");
        if !safe_visible_import_module(base) || names_part.contains('*') {
            return None;
        }
        let names = names_part
            .split(',')
            .map(str::trim)
            .filter(|name| {
                !name.is_empty()
                    && name
                        .chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | ' '))
            })
            .take(12)
            .collect::<Vec<_>>();
        if names.is_empty() {
            return None;
        }
        return Some(format!("from {module} import {}", names.join(", ")));
    }
    None
}

pub(in crate::code_lm_closure) fn safe_visible_import_module(module: &str) -> bool {
    matches!(
        module,
        "abc"
            | "argparse"
            | "base64"
            | "bisect"
            | "calendar"
            | "collections"
            | "configparser"
            | "contextlib"
            | "copy"
            | "csv"
            | "datetime"
            | "decimal"
            | "functools"
            | "glob"
            | "gzip"
            | "hashlib"
            | "heapq"
            | "html"
            | "io"
            | "itertools"
            | "json"
            | "math"
            | "operator"
            | "os"
            | "pathlib"
            | "pickle"
            | "platform"
            | "psutil"
            | "random"
            | "re"
            | "shutil"
            | "statistics"
            | "string"
            | "subprocess"
            | "tarfile"
            | "tempfile"
            | "time"
            | "typing"
            | "urllib"
            | "uuid"
            | "zipfile"
            | "zlib"
    )
}

pub(in crate::code_lm_closure) fn primary_arg_kind(task: &CodeTask) -> ValueKind {
    visible_arg_kinds(&task.entry_point, &task.prompt)
        .into_iter()
        .next()
        .unwrap_or(ValueKind::Unknown)
}

pub(in crate::code_lm_closure) fn visible_arg_kinds(entry: &str, prompt: &str) -> Vec<ValueKind> {
    let mut fallback = Vec::new();
    for line in prompt.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("def ") {
            continue;
        }
        let Some(open) = trimmed.find('(') else {
            continue;
        };
        let name = sanitize_ident(trimmed[4..open].trim());
        let Some(close_rel) = trimmed[open + 1..].find(')') else {
            continue;
        };
        let close = close_rel + open + 1;
        let raw_args = &trimmed[open + 1..close];
        let kinds = raw_args
            .split(',')
            .filter_map(|raw| {
                let item = raw.trim();
                if item.is_empty()
                    || matches!(item, "/" | "*")
                    || item.starts_with('*')
                    || item.starts_with("**")
                {
                    return None;
                }
                Some(annotation_kind(item))
            })
            .collect::<Vec<_>>();
        if name == sanitize_ident(entry) {
            return kinds;
        }
        if fallback.is_empty() {
            fallback = kinds;
        }
    }
    fallback
}

pub(in crate::code_lm_closure) fn annotation_kind(raw_arg: &str) -> ValueKind {
    let lowered = raw_arg.to_lowercase();
    let annotation = lowered
        .split(':')
        .nth(1)
        .unwrap_or("")
        .split('=')
        .next()
        .unwrap_or("")
        .trim();
    if annotation.contains("int") {
        ValueKind::Int
    } else if annotation.contains("list")
        || annotation.contains("tuple")
        || annotation.contains("sequence")
        || annotation.contains("iterable")
    {
        ValueKind::List
    } else if annotation.contains("str") {
        ValueKind::Str
    } else if annotation.contains("dict") || annotation.contains("mapping") {
        ValueKind::Dict
    } else if annotation.contains("bool") {
        ValueKind::Bool
    } else {
        ValueKind::Unknown
    }
}

#[derive(Debug, Clone)]
pub(in crate::code_lm_closure) struct VisibleSignature {
    pub(in crate::code_lm_closure) header: String,
    pub(in crate::code_lm_closure) arg_names: Vec<String>,
    pub(in crate::code_lm_closure) uses_varargs: bool,
}

impl VisibleSignature {
    pub(in crate::code_lm_closure) fn alias_lines(&self) -> Vec<String> {
        if self.uses_varargs {
            return vec![
                "data = args[0] if len(args) > 0 else None".to_string(),
                "other = args[1] if len(args) > 1 else None".to_string(),
                "extra = args[2:] if len(args) > 2 else ()".to_string(),
            ];
        }
        let tuple_expr = match self.arg_names.len() {
            0 => "()".to_string(),
            1 => format!("({},)", self.arg_names[0]),
            _ => format!("({})", self.arg_names.join(", ")),
        };
        let data_expr = self
            .arg_names
            .first()
            .cloned()
            .unwrap_or_else(|| "None".to_string());
        let other_expr = self
            .arg_names
            .get(1)
            .cloned()
            .unwrap_or_else(|| "None".to_string());
        let extra_expr = if self.arg_names.len() > 2 {
            if self.arg_names.len() == 3 {
                format!("({},)", self.arg_names[2])
            } else {
                format!("({})", self.arg_names[2..].join(", "))
            }
        } else {
            "()".to_string()
        };
        vec![
            format!("args = {tuple_expr}"),
            format!("data = {data_expr}"),
            format!("other = {other_expr}"),
            format!("extra = {extra_expr}"),
        ]
    }
}

pub(in crate::code_lm_closure) fn visible_signature(entry: &str, prompt: &str) -> VisibleSignature {
    if let Some((args, arg_names, uses_varargs)) = parse_visible_signature(entry, prompt) {
        return VisibleSignature {
            header: format!("def {entry}({args}):"),
            arg_names,
            uses_varargs,
        };
    }
    VisibleSignature {
        header: format!("def {entry}(*args):"),
        arg_names: Vec::new(),
        uses_varargs: true,
    }
}

pub(in crate::code_lm_closure) fn parse_visible_signature(
    entry: &str,
    prompt: &str,
) -> Option<(String, Vec<String>, bool)> {
    let mut fallback: Option<(String, Vec<String>, bool)> = None;
    for line in prompt.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("def ") {
            continue;
        }
        let open = trimmed.find('(')?;
        let name = sanitize_ident(trimmed[4..open].trim());
        let close = trimmed[open + 1..].find(')')? + open + 1;
        let raw_args = &trimmed[open + 1..close];
        let parsed = sanitize_signature_args(raw_args);
        if name == entry {
            return Some(parsed);
        }
        fallback.get_or_insert(parsed);
    }
    fallback
}

pub(in crate::code_lm_closure) fn sanitize_signature_args(
    raw_args: &str,
) -> (String, Vec<String>, bool) {
    let mut signature_args = Vec::new();
    let mut arg_names = Vec::new();
    let mut uses_varargs = false;
    let mut seen = BTreeSet::new();
    for raw in split_signature_args(raw_args) {
        let mut item = raw.trim();
        if item.is_empty() || matches!(item, "/" | "*") {
            continue;
        }
        if item.starts_with("**") {
            continue;
        }
        let is_vararg = item.starts_with('*');
        if is_vararg {
            item = item.trim_start_matches('*').trim();
        }
        let (name_part, default_part) = split_top_level_once(item, '=')
            .map(|(left, right)| (left.trim(), Some(right.trim())))
            .unwrap_or((item, None));
        let before_type = name_part.split(':').next().unwrap_or(name_part);
        let name = sanitize_ident(before_type.trim());
        if name.is_empty() || !seen.insert(name.clone()) {
            continue;
        }
        if is_vararg {
            uses_varargs = true;
            signature_args.push(format!("*{name}"));
        } else {
            if let Some(default) = default_part.and_then(safe_signature_default_literal) {
                signature_args.push(format!("{name}={default}"));
            } else {
                signature_args.push(name.clone());
            }
            arg_names.push(name);
        }
    }
    if signature_args.is_empty() {
        return ("".to_string(), Vec::new(), false);
    }
    let only_varargs = uses_varargs && arg_names.is_empty();
    (signature_args.join(", "), arg_names, only_varargs)
}

pub(in crate::code_lm_closure) fn split_signature_args(raw_args: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for ch in raw_args.chars() {
        if let Some(q) = quote {
            current.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == q {
                quote = None;
            }
            continue;
        }
        match ch {
            '\'' | '"' => {
                quote = Some(ch);
                current.push(ch);
            }
            '(' | '[' | '{' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' | '}' => {
                depth = (depth - 1).max(0);
                current.push(ch);
            }
            ',' if depth == 0 => {
                out.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

pub(in crate::code_lm_closure) fn split_top_level_once(
    raw: &str,
    target: char,
) -> Option<(&str, &str)> {
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (idx, ch) in raw.char_indices() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == q {
                quote = None;
            }
            continue;
        }
        match ch {
            '\'' | '"' => quote = Some(ch),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = (depth - 1).max(0),
            _ if ch == target && depth == 0 => {
                return Some((&raw[..idx], &raw[idx + ch.len_utf8()..]));
            }
            _ => {}
        }
    }
    None
}

pub(in crate::code_lm_closure) fn safe_signature_default_literal(raw: &str) -> Option<String> {
    let default = raw.trim();
    if default.is_empty() || default.len() > 160 {
        return None;
    }
    let lowered = default.to_lowercase();
    if lowered.contains("__")
        || lowered.contains("lambda")
        || lowered.contains("import")
        || lowered.contains("open(")
        || lowered.contains("exec")
        || lowered.contains("eval")
    {
        return None;
    }
    if default.starts_with('"') || default.starts_with('\'') {
        let quote = default.chars().next().unwrap();
        if default.ends_with(quote) && default.len() >= 2 {
            return Some(default.to_string());
        }
        return None;
    }
    if matches!(default, "None" | "True" | "False") || default.parse::<f64>().is_ok() {
        return Some(default.to_string());
    }
    let bracketed_literal = (default.starts_with('[') && default.ends_with(']'))
        || (default.starts_with('(') && default.ends_with(')'))
        || (default.starts_with('{') && default.ends_with('}'));
    if bracketed_literal
        && default.chars().all(|ch| {
            ch.is_ascii_alphanumeric()
                || ch.is_ascii_whitespace()
                || matches!(
                    ch,
                    '_' | '['
                        | ']'
                        | '('
                        | ')'
                        | '{'
                        | '}'
                        | ','
                        | ':'
                        | '.'
                        | '-'
                        | '+'
                        | '\''
                        | '"'
                )
        })
    {
        return Some(default.to_string());
    }
    None
}
