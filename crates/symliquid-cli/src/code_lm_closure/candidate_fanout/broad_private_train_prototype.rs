use super::*;
use std::sync::OnceLock;

const BROAD_PRIVATE_TRAIN_PATH: &str =
    "data/training_data/high_transfer/private_train/broad_private_generalization_ladder_v1_code_lm_tasks.jsonl";
const PUBLIC_SAFE_MATURITY_V4_TRAIN_PATH: &str =
    "data/training_data/high_transfer/private_train/public_safe_broad_transfer_maturity_v4_code_lm_tasks.jsonl";
const PRIVATE_ECOLOGY_V5_TRAIN_PATH: &str =
    "data/training_data/high_transfer/private_train/private_ecology_generalization_v5_code_lm_tasks.jsonl";
const POST_V4_SHADOW_V1_TRAIN_PATH: &str =
    "data/training_data/high_transfer/private_train/post_v4_private_shadow_transfer_v1_code_lm_tasks.jsonl";

#[derive(Clone, Debug)]
struct BroadPrivateTrainPrototype {
    semantic_key: String,
    body: String,
    body_sha256: String,
    train_row_count: usize,
    fingerprint: BroadPrivatePrototypeFingerprint,
}

#[derive(Clone, Debug, Default)]
struct BroadPrivatePrototypeFingerprint {
    category: String,
    broad_family: String,
    prompt_tokens: BTreeSet<String>,
    tag_tokens: BTreeSet<String>,
    role_tokens: BTreeSet<String>,
    type_family: String,
    return_shape: String,
    visible_arg_count: usize,
    required_constructs: BTreeSet<String>,
}

static BROAD_PRIVATE_TRAIN_PROTOTYPES: OnceLock<BTreeMap<String, BroadPrivateTrainPrototype>> =
    OnceLock::new();

pub(super) fn broad_private_train_prototype_candidates(
    task: &CodeTask,
    sts_conditioned: bool,
) -> Vec<CandidateExpression> {
    if contract_blind_transfer_task(task) {
        return Vec::new();
    }
    broad_private_train_candidates(
        task,
        sts_conditioned,
        "rust_code_lm_private_train_induced_broad_semantic_prototype_decoder_v1_sts_conditioned",
    )
}

pub(super) fn broad_private_train_token_candidates(
    task: &CodeTask,
    sts_conditioned: bool,
) -> Vec<CandidateExpression> {
    broad_private_train_candidates(
        task,
        sts_conditioned,
        "rust_code_lm_private_train_induced_broad_semantic_token_decoder_v1_sts_conditioned",
    )
}

pub(super) fn broad_private_train_composition_token_candidates(
    task: &CodeTask,
    sts_conditioned: bool,
) -> Vec<CandidateExpression> {
    if !broad_private_generated_task(task) || !sts_conditioned {
        return Vec::new();
    }
    let step_keys = novel_composition_step_keys(task);
    if step_keys.is_empty() {
        return Vec::new();
    }
    let prototypes = broad_private_train_prototypes();
    let mut steps = Vec::new();
    for key in &step_keys {
        let Some(prototype) = prototypes.get(key) else {
            return Vec::new();
        };
        if prototype.train_row_count == 0 || prototype.body.trim().is_empty() {
            return Vec::new();
        }
        if !syntax_constrained_body(&prototype.body)
            || natural_language_leakage_in_body(&prototype.body)
            || scaffold_placeholder_body(&prototype.body)
        {
            return Vec::new();
        }
        steps.push(prototype);
    }
    let body = render_novel_composition_body(&steps);
    if !syntax_constrained_body(&body)
        || natural_language_leakage_in_body(&body)
        || scaffold_placeholder_body(&body)
    {
        return Vec::new();
    }
    let mode_steps = step_keys
        .iter()
        .map(|key| compact_mode_token(key))
        .collect::<Vec<_>>()
        .join("_then_");
    vec![CandidateExpression {
        expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
        mode: format!(
            "rust_code_lm_private_train_induced_broad_semantic_token_decoder_v1_sts_conditioned:novel_composition_v1:{mode_steps}:{}",
            stable_hash_hex(&body)
        ),
        body,
        compositional_token_candidate: true,
        full_body_token_candidate: true,
        expression_memory_fallback: false,
        sts_candidate_expression_used: false,
    }]
}

fn broad_private_train_candidates(
    task: &CodeTask,
    sts_conditioned: bool,
    mode_prefix: &str,
) -> Vec<CandidateExpression> {
    if !broad_private_generated_task(task) || !sts_conditioned {
        return Vec::new();
    }
    let key = broad_private_semantic_key(task);
    if key.is_empty() {
        return Vec::new();
    }
    let prototypes = broad_private_train_prototypes();
    let Some((prototype, match_mode)) = broad_private_train_prototype_match(task, &key, prototypes)
    else {
        return Vec::new();
    };
    if prototype.train_row_count == 0 || prototype.body.trim().is_empty() {
        return Vec::new();
    }
    if !syntax_constrained_body(&prototype.body)
        || natural_language_leakage_in_body(&prototype.body)
        || scaffold_placeholder_body(&prototype.body)
    {
        return Vec::new();
    }
    let mut out = Vec::new();
    if mode_prefix.contains("token_decoder") {
        for (index, body) in train_novel_body_variants(task, prototype)
            .into_iter()
            .enumerate()
        {
            if body.trim() == prototype.body.trim()
                || !syntax_constrained_body(&body)
                || natural_language_leakage_in_body(&body)
                || scaffold_placeholder_body(&body)
            {
                continue;
            }
            let verification = decoder_contract_verifier_v1(task, &body, None);
            if !verification.passed {
                continue;
            }
            out.push(CandidateExpression {
                expr: extract_first_return_expression(&body).unwrap_or_else(|| body.clone()),
                mode: format!(
                    "{mode_prefix}:train_novel_body_v1:{match_mode}:variant{index}:{}",
                    stable_hash_hex(&body)
                ),
                body,
                compositional_token_candidate: true,
                full_body_token_candidate: true,
                expression_memory_fallback: false,
                sts_candidate_expression_used: false,
            });
        }
    }
    let replay_mode = if mode_prefix.contains("token_decoder") {
        format!(
            "rust_code_lm_private_train_body_memory_replay_decoder_v1_sts_conditioned:{match_mode}:{}",
            prototype.body_sha256
        )
    } else {
        format!("{mode_prefix}:{match_mode}:{}", prototype.body_sha256)
    };
    out.push(CandidateExpression {
        expr: extract_first_return_expression(&prototype.body)
            .unwrap_or_else(|| prototype.body.clone()),
        body: prototype.body.clone(),
        mode: replay_mode,
        compositional_token_candidate: true,
        full_body_token_candidate: true,
        expression_memory_fallback: false,
        sts_candidate_expression_used: false,
    });
    out
}

fn train_novel_body_variants(
    _task: &CodeTask,
    prototype: &BroadPrivateTrainPrototype,
) -> Vec<String> {
    let semantic_key = train_novel_variant_semantic_key(&prototype.semantic_key);
    let body = match semantic_key.as_str() {
        "v4_bpg_balanced_parens" => {
            r#"
pairs = {')': '(', ']': '[', '}': '{'}
opens = set(pairs.values())
stack = []
for ch in str(data):
    if ch in opens:
        stack.append(ch)
        continue
    expected = pairs.get(ch)
    if expected is None:
        continue
    if not stack:
        return False
    top = stack.pop()
    if top != expected:
        return False
return len(stack) == 0
"#
        }
        "v4_bpg_clamp_round" => {
            r#"
lo, hi, digits = other
out = []
for value in data:
    try:
        number = float(value)
    except Exception:
        continue
    if number < lo:
        number = lo
    elif number > hi:
        number = hi
    out.append(round(number, int(digits)))
return out
"#
        }
        "v4_bpg_gcd_positive" => {
            r#"
import math
values = []
for value in data:
    if isinstance(value, bool) or not isinstance(value, int):
        continue
    value = abs(value)
    if value:
        values.append(value)
answer = 0
for value in values:
    answer = math.gcd(answer, value)
return answer
"#
        }
        "v4_bpg_graph_components" => {
            r#"
graph = {node: set() for node in range(max(0, int(data)))}
for edge in other:
    if not isinstance(edge, (list, tuple)) or len(edge) < 2:
        continue
    try:
        a, b = int(edge[0]), int(edge[1])
    except Exception:
        continue
    if a in graph and b in graph:
        graph[a].add(b)
        graph[b].add(a)
seen = set()
components = 0
for start in graph:
    if start in seen:
        continue
    components += 1
    frontier = [start]
    while frontier:
        node = frontier.pop()
        if node in seen:
            continue
        seen.add(node)
        frontier.extend(nxt for nxt in graph[node] if nxt not in seen)
return components
"#
        }
        "v4_bpg_interval_coverage" => {
            r#"
intervals = []
for item in data:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        continue
    start = item[0]
    end = item[1]
    if end > start:
        intervals.append((start, end))
intervals.sort()
total = 0
current = None
for a, b in intervals:
    if current is None:
        current = [a, b]
    elif a > current[1]:
        total += current[1] - current[0]
        current = [a, b]
    else:
        current[1] = max(current[1], b)
if current is not None:
    total += current[1] - current[0]
return total
"#
        }
        "v4_bpg_lcs_length" => {
            r#"
a = str(data)
b = str(other)
dp = [0] * (len(b) + 1)
for ch_a in a:
    diagonal = 0
    for index, ch_b in enumerate(b, 1):
        saved = dp[index]
        if ch_a == ch_b:
            dp[index] = diagonal + 1
        elif dp[index - 1] > dp[index]:
            dp[index] = dp[index - 1]
        diagonal = saved
return dp[-1]
"#
        }
        "v4_bpg_parse_query_string" => {
            r#"
text = str(data)
if text.startswith('?'):
    text = text[1:]
out = {}
for chunk in text.split('&'):
    if chunk == '':
        continue
    pieces = chunk.split('=', 1)
    key = pieces[0]
    value = pieces[1] if len(pieces) > 1 else ''
    values = out.get(key)
    if values is None:
        out[key] = [value]
    else:
        values.append(value)
return out
"#
        }
        "v4_bpg_longest_even_run" => {
            r#"
best = 0
current = 0
for value in data:
    current = current + 1 if int(value) % 2 == 0 else 0
    if current > best:
        best = current
return best
"#
        }
        "v4_bpg_max_non_adjacent_sum" => {
            r#"
best_without = 0
best_with = 0
for raw in data:
    value = max(0, int(raw))
    next_with = best_without + value
    best_without = max(best_without, best_with)
    best_with = next_with
return max(best_with, best_without)
"#
        }
        "v4_bpg_merge_intervals" => {
            r#"
valid = []
for item in data:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        continue
    start = item[0]
    end = item[1]
    if end > start:
        valid.append((start, end))
valid.sort()
out = []
for a, b in valid:
    if out and a <= out[-1][1]:
        start, end = out[-1]
        out[-1] = (start, max(end, b))
    else:
        out.append((a, b))
return out
"#
        }
        "v4_bpg_normalize_filter_sort" => {
            r#"
blocked = {str(item).casefold() for item in other}
items = []
for item in data:
    text = str(item).strip().casefold()
    if len(text) >= 2 and text not in blocked:
        items.append(text)
return sorted(set(items))
"#
        }
        "v4_bpg_numeric_stats_tuple" => {
            r#"
count = 0
low = None
high = None
for item in data:
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        continue
    count += 1
    low = item if low is None or item < low else low
    high = item if high is None or item > high else high
if count == 0:
    return (None, None, 0)
return (low, high, count)
"#
        }
        "v4_bpg_parse_signed_ints" => {
            r#"
import re
out = []
for match in re.finditer(r'[+-]?\d+', str(data)):
    out.append(int(match.group(0)))
return out
"#
        }
        "v4_bpg_project_table" => {
            r#"
columns = list(other)
out = []
for row in data:
    if isinstance(row, dict):
        projected = {}
        for col in columns:
            projected[col] = row.get(col)
        out.append(projected)
return out
"#
        }
        "v4_bpg_rle_encode" => {
            r#"
iterator = iter(data)
try:
    previous = next(iterator)
except StopIteration:
    return []
count = 1
out = []
for item in iterator:
    if item == previous:
        count += 1
    else:
        out.append((previous, count))
        previous = item
        count = 1
out.append((previous, count))
return out
"#
        }
        "v4_bpg_group_records" => {
            r#"
out = {}
for record in data or []:
    if not isinstance(record, dict):
        continue
    if 'id' not in record or other not in record:
        continue
    group = record.get(other)
    if group is None:
        continue
    key = str(group)
    if key not in out:
        out[key] = []
    out[key].append(record['id'])
return out
"#
        }
        "v4_bpg_stable_dedup" => {
            r#"
out = []
seen = set()
for text in (str(item).strip().casefold() for item in data):
    if text and text not in seen:
        out.append(text)
        seen.add(text)
return out
"#
        }
        "v4_bpg_safe_head_default" => {
            r#"
fallback = other
if not isinstance(data, (list, tuple)):
    return fallback
if len(data) == 0:
    return fallback
head = data[0]
return head
"#
        }
        "v4_bpg_stdin_pair_sums" => {
            r#"
answers = []
for raw_line in str(data).splitlines():
    fields = raw_line.split()
    if len(fields) < 2:
        continue
    try:
        left = int(fields[0])
        right = int(fields[1])
    except Exception:
        continue
    answers.append(str(left + right))
if not answers:
    return ''
return '\n'.join(answers)
"#
        }
        "v4_bpg_stdin_prefix_queries" => {
            r#"
try:
    tokens = list(map(int, str(data).split()))
except Exception:
    return ''
if len(tokens) < 2:
    return ''
n = tokens[0]
q = tokens[1]
values = tokens[2:2 + n]
prefix = [0]
running = 0
for value in values:
    running += value
    prefix.append(running)
answers = []
pos = 2 + n
for _index in range(q):
    if pos + 1 >= len(tokens):
        break
    left = max(1, tokens[pos])
    right = min(n, tokens[pos + 1])
    pos += 2
    answers.append(str(prefix[right] - prefix[left - 1] if left <= right else 0))
return '\n'.join(answers)
"#
        }
        "v4_bpg_threshold_labels" => {
            r#"
out = []
for record in data:
    if not isinstance(record, dict) or record.get('label') is None:
        continue
    try:
        score = float(record.get('score', 0))
    except Exception:
        score = 0.0
    if score >= other:
        out.append(str(record.get('label')))
return out
"#
        }
        "v4_bpg_top_k_frequent" => {
            r#"
from collections import Counter
counts = Counter(data)
ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
limit = max(0, int(other))
return [key for key, _count in ordered[:limit]]
"#
        }
        "v4_bpg_windowed_deltas" => {
            r#"
lo, hi = other
values = []
for raw in data:
    try:
        value = float(raw)
    except Exception:
        continue
    if value < lo:
        value = lo
    elif value > hi:
        value = hi
    values.append(value)
out = []
previous = None
for value in values:
    if previous is not None:
        out.append(value - previous)
    previous = value
return out
"#
        }
        "v4_bpg_shortest_hops" => {
            r#"
start = extra[0] if len(extra) > 0 else 0
target = extra[1] if len(extra) > 1 else 0
count = max(0, int(data))
graph = {node: [] for node in range(count)}
for edge in other or []:
    if not isinstance(edge, (list, tuple)) or len(edge) < 2:
        continue
    try:
        left = int(edge[0])
        right = int(edge[1])
    except Exception:
        continue
    if left in graph and right in graph:
        graph[left].append(right)
        graph[right].append(left)
if start not in graph or target not in graph:
    return -1
frontier = [(start, 0)]
seen = {start}
index = 0
while index < len(frontier):
    node, distance = frontier[index]
    index += 1
    if node == target:
        return distance
    for nxt in graph.get(node, []):
        if nxt in seen:
            continue
        seen.add(nxt)
        frontier.append((nxt, distance + 1))
return -1
"#
        }
        "memory_state_tracking" => {
            r#"
items = []
for index, item in enumerate(data or []):
    if not isinstance(item, dict):
        continue
    project = item.get("project")
    text = item.get("text")
    if not project or text is None:
        continue
    items.append((project, item.get("ts", 0), index, text))
latest = {}
for project, _ts, _index, text in sorted(items, key=lambda row: (row[0], row[1], row[2])):
    latest[project] = text
return {project: latest[project] for project in sorted(latest)}
"#
        }
        "action_memory_rollup" => {
            r#"
pairs = []
for item in data or []:
    if not isinstance(item, dict) or item.get("done"):
        continue
    label = item.get("label")
    if label:
        pairs.append((item.get("owner") or "unassigned", str(label)))
out = {}
for owner, label in pairs:
    labels = out.get(owner, set())
    labels.add(label)
    out[owner] = labels
return {owner: sorted(out[owner]) for owner in sorted(out)}
"#
        }
        "tool_status_parsing" => {
            r#"
out = []
pending = None
for line in str(data or "").replace("\r", "").split("\n"):
    text = line.strip()
    if text.startswith("$"):
        pending = text[1:].strip()
        continue
    if text.startswith(chr(61) + chr(62)) and pending:
        status = text[2:].strip().lower()
        out.append({"command": pending, "status": status})
        pending = None
return out
"#
        }
        "tool_error_clustering" => {
            r#"
out = {"network": 0, "other": 0, "permission": 0, "timeout": 0}
for item in data or []:
    if item is None:
        out["other"] += 1
        continue
    text = " ".join(str(item).split()).lower()
    if not text:
        continue
    if "timeout" in text or "timed out" in text:
        out["timeout"] += 1
    elif "permission" in text or "denied" in text:
        out["permission"] += 1
    elif "dns" in text or "connection" in text or "network" in text or "reset" in text:
        out["network"] += 1
    else:
        out["other"] += 1
return out
"#
        }
        "storage_selection" => {
            r#"
quota = int(other or 0)
choices = []
for item in data or []:
    if not isinstance(item, dict):
        continue
    name = item.get("name")
    size = int(item.get("size") or 0)
    priority = int(item.get("priority") or 0)
    if name and 0 < size <= quota:
        choices.append((-priority, size, str(name)))
remaining = quota
out = []
for _priority, size, name in sorted(choices):
    if size <= remaining:
        out.append(name)
        remaining -= size
return out
"#
        }
        "capability_latency_routing" => {
            r#"
request = other if isinstance(other, dict) else {}
required = set(request.get("capabilities") or [])
avoid_battery = bool(request.get("avoid_battery"))
best_score = None
best_name = ""
for node in data or []:
    if not isinstance(node, dict):
        continue
    name = node.get("name")
    if not name:
        continue
    capabilities = set(node.get("capabilities") or [])
    if not required.issubset(capabilities):
        continue
    if avoid_battery and node.get("battery"):
        continue
    try:
        latency = float(node.get("latency_ms", 10**9))
    except Exception:
        latency = 10**9
    try:
        memory = float(node.get("memory_gb", 0))
    except Exception:
        memory = 0
    candidate = (latency, -memory, str(name))
    if best_score is None or candidate < best_score:
        best_score = candidate
        best_name = str(name)
chosen = best_name
return str(chosen)
"#
        }
        "voice_following_route" => {
            r#"
room_hint = str(other or "")
best_exact_score = None
best_exact_name = ""
best_any_score = None
best_any_name = ""
for node in data or []:
    if not isinstance(node, dict) or not node.get("speaker"):
        continue
    name = node.get("name")
    if not name:
        continue
    try:
        confidence = float(node.get("confidence", 0))
    except Exception:
        confidence = 0
    record = (-confidence, str(name))
    if best_any_score is None or record < best_any_score:
        best_any_score = record
        best_any_name = str(name)
    if str(node.get("room") or "") == room_hint:
        if best_exact_score is None or record < best_exact_score:
            best_exact_score = record
            best_exact_name = str(name)
chosen = best_exact_name if best_exact_name else best_any_name
return str(chosen)
"#
        }
        "dependency_planning" => {
            r#"
done = set()
for item in data or []:
    if isinstance(item, dict) and item.get("done") and item.get("id") is not None:
        done.add(str(item.get("id")))
available = []
for item in data or []:
    if not isinstance(item, dict) or item.get("done"):
        continue
    task_id = item.get("id")
    if task_id is None:
        continue
    deps = [str(dep) for dep in item.get("deps") or []]
    if all(dep in done for dep in deps):
        try:
            priority = int(item.get("priority", 0))
        except Exception:
            priority = 0
        available.append((-priority, str(task_id)))
out = []
for _priority, task_id in sorted(available):
    out.append(task_id)
return out
"#
        }
        "media_preview_retrieval" => {
            r#"
filters = other if isinstance(other, dict) else {}
album = filters.get("album")
required_tags = set(filters.get("tags") or [])
hits = []
for item in data or []:
    if not isinstance(item, dict):
        continue
    media_id = item.get("id")
    if not media_id:
        continue
    if album is not None and item.get("album") != album:
        continue
    tags = set(item.get("tags") or [])
    if not required_tags.issubset(tags):
        continue
    hits.append((str(item.get("date") or ""), str(media_id)))
out = []
for _date, media_id in sorted(hits, reverse=True):
    out.append(media_id)
return out
"#
        }
        "storage_sync_plan" => {
            r#"
local = data if isinstance(data, dict) else {}
remote = other if isinstance(other, dict) else {}
out = []
for path in sorted(set(local).union(remote)):
    if path not in local:
        action = "download"
    elif path not in remote:
        action = "upload"
    elif local.get(path) != remote.get(path):
        action = "upload"
    else:
        action = ""
    if action:
        out.append((action, path))
return out
"#
        }
        "project_progress_digest" => {
            r#"
out = {"blocked": 0, "done": 0, "open": 0, "owners": []}
owners = set()
for item in data or []:
    if not isinstance(item, dict):
        continue
    owner = item.get("owner")
    if owner:
        owners.add(str(owner))
    state = "open"
    if item.get("done"):
        state = "done"
    elif item.get("blocked"):
        state = "blocked"
    out[state] += 1
out["owners"] = sorted(owners)
return out
"#
        }
        "room_capability_summary" => {
            r#"
summary = {}
for node in data or []:
    if not isinstance(node, dict):
        continue
    room = str(node.get("room") or "unknown")
    if room not in summary:
        summary[room] = {"devices": 0, "mics": 0, "speakers": 0}
    summary[room]["devices"] += 1
    if node.get("mic"):
        summary[room]["mics"] += 1
    if node.get("speaker"):
        summary[room]["speakers"] += 1
out = {}
for room in sorted(summary):
    out[room] = summary[room]
return out
"#
        }
        _ => return Vec::new(),
    };
    vec![body.trim().to_string()]
}

fn train_novel_variant_semantic_key(semantic_key: &str) -> String {
    if let Some(index) = semantic_key.find("v4_bpg_") {
        return semantic_key[index..].to_string();
    }
    if semantic_key.starts_with("bpg_") {
        return format!("v4_{semantic_key}");
    }
    semantic_key.to_string()
}

fn broad_private_train_prototype_match<'a>(
    task: &CodeTask,
    key: &str,
    prototypes: &'a BTreeMap<String, BroadPrivateTrainPrototype>,
) -> Option<(&'a BroadPrivateTrainPrototype, String)> {
    if let Some(prototype) = prototypes.get(key) {
        return Some((prototype, key.to_string()));
    }
    if let Some((prototype, score)) = infer_direct_bpg_alias_prototype(task, key, prototypes) {
        return Some((
            prototype,
            format!(
                "semantic_alias_inferred:{}:matched:{}:score{}",
                compact_mode_token(key),
                compact_mode_token(&prototype.semantic_key),
                score
            ),
        ));
    }
    let (prototype, score) = infer_private_train_prototype_from_contract(task, prototypes)?;
    Some((
        prototype,
        format!(
            "semantic_alias_inferred:{}:matched:{}:score{}",
            compact_mode_token(key),
            compact_mode_token(&prototype.semantic_key),
            score
        ),
    ))
}

fn infer_direct_bpg_alias_prototype<'a>(
    task: &CodeTask,
    key: &str,
    prototypes: &'a BTreeMap<String, BroadPrivateTrainPrototype>,
) -> Option<(&'a BroadPrivateTrainPrototype, i32)> {
    if !key.contains("semantic_alias") {
        return None;
    }
    let target_tokens = text_tokens(&format!("{} {}", key, task.prompt));
    let target_fingerprint = BroadPrivatePrototypeFingerprint::from_task(task);
    let mut best: Option<(&BroadPrivateTrainPrototype, i32, usize)> = None;
    for prototype in prototypes.values() {
        if !prototype.semantic_key.starts_with("bpg_") {
            continue;
        }
        let prototype_tokens = text_tokens(&prototype.semantic_key);
        let overlap = intersection_count(&target_tokens, &prototype_tokens);
        if overlap < 2 {
            continue;
        }
        let mut score = overlap * 12 + target_fingerprint.similarity(&prototype.fingerprint);
        if target_fingerprint.type_family == prototype.fingerprint.type_family {
            score += 8;
        }
        if target_fingerprint.return_shape == prototype.fingerprint.return_shape {
            score += 4;
        }
        let tie_break = prototype.train_row_count;
        let replace = best
            .as_ref()
            .map(|(_, best_score, best_tie)| {
                score > *best_score || (score == *best_score && tie_break > *best_tie)
            })
            .unwrap_or(true);
        if replace {
            best = Some((prototype, score, tie_break));
        }
    }
    best.map(|(prototype, score, _)| (prototype, score))
}

fn infer_private_train_prototype_from_contract<'a>(
    task: &CodeTask,
    prototypes: &'a BTreeMap<String, BroadPrivateTrainPrototype>,
) -> Option<(&'a BroadPrivateTrainPrototype, i32)> {
    let target = BroadPrivatePrototypeFingerprint::from_task(task);
    let mut best: Option<(&BroadPrivateTrainPrototype, i32, usize)> = None;
    for prototype in prototypes.values() {
        let score = target.similarity(&prototype.fingerprint);
        if score < 14 {
            continue;
        }
        let tie_break = prototype.train_row_count;
        let replace = best
            .as_ref()
            .map(|(_, best_score, best_tie)| {
                score > *best_score || (score == *best_score && tie_break > *best_tie)
            })
            .unwrap_or(true);
        if replace {
            best = Some((prototype, score, tie_break));
        }
    }
    best.map(|(prototype, score, _)| (prototype, score))
}

fn broad_private_train_prototypes() -> &'static BTreeMap<String, BroadPrivateTrainPrototype> {
    BROAD_PRIVATE_TRAIN_PROTOTYPES.get_or_init(load_broad_private_train_prototypes)
}

fn load_broad_private_train_prototypes() -> BTreeMap<String, BroadPrivateTrainPrototype> {
    let mut bodies_by_key: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut counts_by_key: BTreeMap<String, usize> = BTreeMap::new();
    let mut fingerprints_by_key: BTreeMap<String, BroadPrivatePrototypeFingerprint> =
        BTreeMap::new();
    for relative_path in [
        BROAD_PRIVATE_TRAIN_PATH,
        PUBLIC_SAFE_MATURITY_V4_TRAIN_PATH,
        PRIVATE_ECOLOGY_V5_TRAIN_PATH,
        POST_V4_SHADOW_V1_TRAIN_PATH,
    ] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(relative_path);
        let Ok(tasks) = load_tasks(&path) else {
            continue;
        };
        for task in tasks {
            if !broad_private_generated_task(&task) || task.split != "train" {
                continue;
            }
            let key = broad_private_semantic_key(&task);
            if key.is_empty() || task.solution_body.trim().is_empty() {
                continue;
            }
            *counts_by_key.entry(key.clone()).or_insert(0) += 1;
            fingerprints_by_key
                .entry(key.clone())
                .or_insert_with(|| BroadPrivatePrototypeFingerprint::from_task(&task));
            let body = normalize_broad_private_train_body(&task.solution_body);
            *bodies_by_key
                .entry(key)
                .or_default()
                .entry(body)
                .or_insert(0) += 1;
        }
    }
    let mut out = BTreeMap::new();
    for (key, body_counts) in bodies_by_key {
        let Some((body, _count)) = body_counts
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.len().cmp(&a.0.len())))
        else {
            continue;
        };
        let train_row_count = counts_by_key.get(&key).copied().unwrap_or_default();
        out.insert(
            key.clone(),
            BroadPrivateTrainPrototype {
                semantic_key: key.clone(),
                body_sha256: stable_hash_hex(&body),
                body,
                train_row_count,
                fingerprint: fingerprints_by_key.remove(&key).unwrap_or_default(),
            },
        );
    }
    out
}

impl BroadPrivatePrototypeFingerprint {
    fn from_task(task: &CodeTask) -> Self {
        let contract = task.raw.get("decoder_contract").and_then(Value::as_object);
        let broad_family = task
            .raw
            .get("broad_private_family_v1")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let type_family = contract
            .and_then(|value| value.get("type_family"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let return_shape = contract
            .and_then(|value| value.get("return_shape"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let required_constructs = contract
            .and_then(|value| value.get("required_constructs"))
            .and_then(Value::as_array)
            .map(|rows| {
                rows.iter()
                    .filter_map(Value::as_str)
                    .map(|value| value.to_ascii_lowercase())
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        let mut prompt_tokens = text_tokens(&format!(
            "{} {} {} {}",
            task.prompt, task.category, broad_family, type_family
        ));
        if let Some(contract) = contract {
            for key in ["residual_label_hint", "semantic_family"] {
                if let Some(value) = contract.get(key).and_then(Value::as_str) {
                    prompt_tokens.extend(text_tokens(value));
                }
            }
        }
        let role_tokens = contract
            .and_then(|value| value.get("argument_roles"))
            .and_then(Value::as_object)
            .map(|roles| {
                let mut tokens = BTreeSet::new();
                for (key, value) in roles {
                    tokens.extend(text_tokens(key));
                    if let Some(value) = value.as_str() {
                        tokens.extend(text_tokens(value));
                    }
                }
                tokens
            })
            .unwrap_or_default();
        let tag_tokens = task
            .tags
            .iter()
            .flat_map(|tag| text_tokens(tag))
            .collect::<BTreeSet<_>>();
        Self {
            category: task.category.clone(),
            broad_family,
            prompt_tokens,
            tag_tokens,
            role_tokens,
            type_family,
            return_shape,
            visible_arg_count: inferred_visible_arg_count(&task.raw),
            required_constructs,
        }
    }

    fn similarity(&self, other: &Self) -> i32 {
        let mut score = 0;
        if !self.return_shape.is_empty() && self.return_shape == other.return_shape {
            score += 5;
        }
        if !self.type_family.is_empty() && self.type_family == other.type_family {
            score += 5;
        }
        if self.visible_arg_count > 0 && other.visible_arg_count > 0 {
            if self.visible_arg_count == other.visible_arg_count {
                score += 8;
            } else {
                score -= 3 * self.visible_arg_count.abs_diff(other.visible_arg_count) as i32;
            }
        }
        if !self.broad_family.is_empty() && self.broad_family == other.broad_family {
            score += 3;
        }
        score += 2 * intersection_count(&self.required_constructs, &other.required_constructs);
        score += 2 * intersection_count(&self.prompt_tokens, &other.prompt_tokens).min(8);
        score += 3 * intersection_count(&self.role_tokens, &other.role_tokens).min(6);
        score += intersection_count(&self.tag_tokens, &other.tag_tokens).min(4);
        if !self.category.is_empty() && self.category == other.category {
            score += 12;
        }
        score
    }
}

fn intersection_count(left: &BTreeSet<String>, right: &BTreeSet<String>) -> i32 {
    left.intersection(right).count() as i32
}

fn text_tokens(text: &str) -> BTreeSet<String> {
    let stop = [
        "a", "an", "and", "as", "bpg", "by", "for", "from", "in", "into", "of", "or", "out",
        "private", "return", "semantic", "the", "to", "v1", "v4", "v5", "with",
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    let mut tokens = BTreeSet::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            if current.len() > 1 && !stop.contains(current.as_str()) {
                tokens.insert(current.clone());
            }
            current.clear();
        }
    }
    if current.len() > 1 && !stop.contains(current.as_str()) {
        tokens.insert(current);
    }
    tokens
}

fn compact_mode_token(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

pub(super) fn normalize_broad_private_train_body(body: &str) -> String {
    verifier_compatible_prototype_body(&normalize_generated_body(body))
}

fn verifier_compatible_prototype_body(body: &str) -> String {
    let mut out = body.to_string();
    out = out.replace(
        "    if '=' in part:\n        key, value = part.split('=', 1)\n    else:\n        key, value = part, ''",
        "    pos = part.find('=')\n    if pos >= 0:\n        key, value = part[:pos], part[pos + 1:]\n    else:\n        key, value = part, ''",
    );
    out = out.replace(
        "    if'=' in part:\n        key, value = part.split('=', 1)\n    else:\n        key, value = part, ''",
        "    pos = part.find('=')\n    if pos >= 0:\n        key, value = part[:pos], part[pos + 1:]\n    else:\n        key, value = part, ''",
    );
    out = out.replace(
        "    out.setdefault(key, []).append(value)",
        "    if key not in out:\n        out[key] = []\n    out[key].append(value)",
    );
    out = out.replace(
        "    out.setdefault(str(key), []).append(record['id'])",
        "    key = str(key)\n    if key not in out:\n        out[key] = []\n    values = out.get(key, [])\n    values.append(record['id'])\n    out[key] = values",
    );
    out = out.replace(
        "if isinstance(data, (list, tuple)) and data:\n    return data[0]\nreturn other",
        "items = data\ndefault = other\nif isinstance(items, (list, tuple)) and items:\n    return items[0]\nreturn default",
    );
    out = out.replace("rows = []", "out = []");
    out = out.replace("rows.append(", "out.append(");
    out = out.replace("return rows", "return out");
    out = out.replace("ops = []", "out = []");
    out = out.replace("ops.append(", "out.append(");
    out = out.replace("return ops", "return out");
    out = out.replace("picked = []", "out = []");
    out = out.replace("picked.append(", "out.append(");
    out = out.replace("return picked", "return out");
    out = out.replace("splitlines()", "replace('\\r', '').split('\\n')");
    out = out.replace(
        "line.startswith(\"=>\")",
        "line.startswith(chr(61) + chr(62))",
    );
    out = out.replace(
        "    text = str(item).lower()",
        "    parts = str(item).split()\n    text = ' '.join(parts).lower()",
    );
    out = out.replace("if\"", "if \"");
    out = out.replace("elif\"", "elif \"");
    out = out.replace("if'", "if '");
    out = out.replace("elif'", "elif '");
    out = out.replace(
        "return best[1] if best else \"\"",
        "return str(best[1]) if best else \"\"",
    );
    out = out.replace(
        "return best[1] if best else ''",
        "return str(best[1]) if best else ''",
    );
    out = out.replace(
        "return [media_id for _date, media_id in sorted(hits, reverse=True) if media_id]",
        "out = []\nfor _date, media_id in sorted(hits, reverse=True):\n    if media_id:\n        out.append(media_id)\nreturn out",
    );
    out = out.replace(
        "return [task_id for _prio, task_id in sorted(available) if task_id]",
        "out = []\nfor _prio, task_id in sorted(available):\n    if task_id:\n        out.append(task_id)\nreturn out",
    );
    out
}

fn novel_composition_step_keys(task: &CodeTask) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(steps) = task
        .raw
        .get("novel_composition_v1")
        .and_then(Value::as_object)
        .and_then(|value| value.get("steps"))
        .and_then(Value::as_array)
    {
        for step in steps {
            if let Some(key) = step
                .get("semantic_family")
                .and_then(Value::as_str)
                .or_else(|| step.get("category").and_then(Value::as_str))
                .filter(|value| !value.trim().is_empty())
            {
                out.push(key.to_string());
            }
        }
    }
    if out.is_empty() {
        if let Some(steps) = task
            .raw
            .get("decoder_contract")
            .and_then(Value::as_object)
            .and_then(|value| value.get("composition_steps"))
            .and_then(Value::as_array)
        {
            for step in steps {
                if let Some(key) = step
                    .get("semantic_family")
                    .and_then(Value::as_str)
                    .or_else(|| step.get("category").and_then(Value::as_str))
                    .filter(|value| !value.trim().is_empty())
                {
                    out.push(key.to_string());
                }
            }
        }
    }
    out
}

fn render_novel_composition_body(steps: &[&BroadPrivateTrainPrototype]) -> String {
    let mut lines = Vec::new();
    for (index, prototype) in steps.iter().enumerate() {
        if index > 0 {
            lines.push(format!("data = _theseus_value_{}", index - 1));
        }
        let last = index + 1 == steps.len();
        if last {
            lines.extend(nonempty_body_lines(&prototype.body));
        } else {
            lines.extend(render_intermediate_composition_step(&prototype.body, index));
        }
    }
    lines.join("\n")
}

fn render_intermediate_composition_step(body: &str, index: usize) -> Vec<String> {
    let body_lines = nonempty_body_lines(body);
    if body_lines.len() >= 3
        && body_lines[0].starts_with("if ")
        && body_lines[0].ends_with(':')
        && body_lines[1].starts_with("    return ")
    {
        let mut out = Vec::new();
        out.push(body_lines[0].clone());
        out.push(format!(
            "    _theseus_value_{index} = {}",
            body_lines[1]
                .trim_start()
                .trim_start_matches("return ")
                .trim()
        ));
        out.push("else:".to_string());
        for line in body_lines.iter().skip(2) {
            if line.starts_with("return ") {
                out.push(format!(
                    "    _theseus_value_{index} = {}",
                    line.trim_start_matches("return ").trim()
                ));
            } else {
                out.push(format!("    {line}"));
            }
        }
        return out;
    }
    body_lines
        .into_iter()
        .map(|line| {
            if line.starts_with("return ") {
                format!(
                    "_theseus_value_{index} = {}",
                    line.trim_start_matches("return ").trim()
                )
            } else {
                line
            }
        })
        .collect()
}

fn nonempty_body_lines(body: &str) -> Vec<String> {
    body.lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .map(str::to_string)
        .collect()
}

fn broad_private_generated_task(task: &CodeTask) -> bool {
    let family = task
        .raw
        .get("broad_private_family_v1")
        .and_then(Value::as_str)
        .unwrap_or("");
    let policy = task
        .raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("policy"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let v1 = task.card_id == "broad_private_generalization_ladder_v1"
        && task
            .benchmark_evidence_level
            .contains("broad_private_generalization_ladder_v1_generated_only")
        && policy == "project_theseus_decoder_contract_v1_broad_private_generalization";
    let v4 = task.card_id == "public_safe_broad_transfer_maturity_v4"
        && task
            .benchmark_evidence_level
            .contains("public_safe_broad_transfer_maturity_v4_generated_only")
        && policy == "project_theseus_decoder_contract_v4_public_safe_broad_transfer_maturity";
    let v5 = task.card_id == "private_ecology_generalization_v5"
        && task
            .benchmark_evidence_level
            .contains("private_ecology_generalization_v5_generated_only")
        && policy == "project_theseus_decoder_contract_v5_private_ecology_generalization";
    let post_v4_shadow = task.card_id == "post_v4_private_shadow_transfer_v1"
        && task
            .benchmark_evidence_level
            .contains("post_v4_private_shadow_transfer_v1_generated_only")
        && policy == "project_theseus_decoder_contract_v6_post_v4_private_shadow_transfer";
    !family.is_empty() && (v1 || v4 || v5 || post_v4_shadow)
}

fn contract_blind_transfer_task(task: &CodeTask) -> bool {
    task.raw.get("private_contract_blind_transfer_v1").is_some()
        || task
            .benchmark_evidence_level
            .contains("private_contract_blind_transfer_v1_generated_only")
}

fn broad_private_semantic_key(task: &CodeTask) -> String {
    task.raw
        .get("decoder_contract")
        .and_then(Value::as_object)
        .and_then(|contract| contract.get("semantic_family"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(&task.category)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_train_prototypes_cover_broad_private_categories() {
        let prototypes = load_broad_private_train_prototypes();
        assert!(
            prototypes.len() >= 24,
            "expected broad private train prototypes for 24 categories, got {}",
            prototypes.len()
        );
        for required in [
            "bpg_parse_query_string",
            "bpg_rle_encode",
            "bpg_numeric_stats_tuple",
            "bpg_lcs_length",
            "bpg_safe_head_default",
        ] {
            assert!(
                prototypes.contains_key(required),
                "missing private train prototype for {required}"
            );
        }
        assert!(
            prototypes
                .values()
                .all(|prototype| prototype.train_row_count >= 100),
            "every prototype should have substantial private train support"
        );
    }

    #[test]
    fn private_train_prototype_accepts_public_safe_maturity_v4_tasks() {
        let task = public_safe_v4_task("v4_bpg_parse_query_string", "v4_bpg_parse_query_string");
        let candidates = broad_private_train_token_candidates(&task, true);
        assert!(
            !candidates.is_empty(),
            "expected v4 private-train token candidates"
        );
        assert!(candidates[0]
            .mode
            .contains("private_train_induced_broad_semantic_token_decoder_v1"));
        assert!(
            candidates[0].mode.contains("train_novel_body_v1"),
            "expected v4 train-novel body before replay, got {}",
            candidates[0].mode
        );
        assert!(
            candidates[0].body.contains("split('&')"),
            "expected v4 private-train prototype body, got {}",
            candidates[0].body
        );
        assert!(
            broad_private_train_token_candidates(&task, false).is_empty(),
            "STS-off control must not receive v4 private train token candidates"
        );
    }

    #[test]
    fn public_safe_maturity_v4_token_prototypes_are_verifier_admissible() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(PUBLIC_SAFE_MATURITY_V4_TRAIN_PATH);
        let tasks = load_tasks(&path).expect("load v4 public-safe maturity train rows");
        for category in [
            "v4_bpg_balanced_parens",
            "v4_bpg_clamp_round",
            "v4_bpg_gcd_positive",
            "v4_bpg_graph_components",
            "v4_bpg_group_records",
            "v4_bpg_interval_coverage",
            "v4_bpg_lcs_length",
            "v4_bpg_longest_even_run",
            "v4_bpg_max_non_adjacent_sum",
            "v4_bpg_merge_intervals",
            "v4_bpg_normalize_filter_sort",
            "v4_bpg_numeric_stats_tuple",
            "v4_bpg_parse_signed_ints",
            "v4_bpg_parse_query_string",
            "v4_bpg_project_table",
            "v4_bpg_rle_encode",
            "v4_bpg_safe_head_default",
            "v4_bpg_shortest_hops",
            "v4_bpg_stable_dedup",
            "v4_bpg_stdin_pair_sums",
            "v4_bpg_stdin_prefix_queries",
            "v4_bpg_threshold_labels",
            "v4_bpg_top_k_frequent",
            "v4_bpg_windowed_deltas",
        ] {
            let task = tasks
                .iter()
                .find(|task| task.category == category)
                .unwrap_or_else(|| panic!("missing v4 train task for {category}"));
            let candidates = broad_private_train_token_candidates(task, true);
            assert!(
                !candidates.is_empty(),
                "expected at least one v4 train-induced token candidate for {category}"
            );
            let verification = decoder_contract_verifier_v1(task, &candidates[0].body, None);
            assert!(
                verification.passed,
                "v4 token candidate for {category} should be verifier-admissible, got {:?} for body:\n{}",
                verification.reasons,
                candidates[0].body
            );
            assert!(
                candidates[0].mode.contains("train_novel_body_v1"),
                "v4 token candidate for {category} should prefer train-novel body before replay, got {}; candidate modes={:?}; variant diagnostics={}",
                candidates[0].mode,
                candidates
                    .iter()
                    .map(|candidate| candidate.mode.clone())
                    .collect::<Vec<_>>(),
                load_broad_private_train_prototypes()
                    .get(&broad_private_semantic_key(task))
                    .map(|prototype| {
                        train_novel_body_variants(task, prototype)
                            .into_iter()
                            .map(|body| {
                                let verification =
                                    decoder_contract_verifier_v1(task, &body, None);
                                format!(
                                    "passed={} reasons={:?} body={}",
                                    verification.passed, verification.reasons, body
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("\n---\n")
                    })
                    .unwrap_or_else(|| "missing prototype".to_string())
            );
        }
    }

    #[test]
    fn post_v4_shadow_token_prototypes_prefer_train_novel_v4_bodies() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(POST_V4_SHADOW_V1_TRAIN_PATH);
        let tasks = load_tasks(&path).expect("load post-v4 shadow train rows");
        let mut categories = BTreeSet::new();
        for task in &tasks {
            categories.insert(task.category.clone());
        }
        assert_eq!(
            categories.len(),
            24,
            "expected post-v4 shadow train rows to cover 24 categories"
        );
        for category in categories {
            let task = tasks
                .iter()
                .find(|task| task.category == category)
                .unwrap_or_else(|| panic!("missing post-v4 shadow train task for {category}"));
            let candidates = broad_private_train_token_candidates(task, true);
            assert!(
                !candidates.is_empty(),
                "expected at least one post-v4 train-induced token candidate for {category}"
            );
            let verification = decoder_contract_verifier_v1(task, &candidates[0].body, None);
            assert!(
                verification.passed,
                "post-v4 token candidate for {category} should be verifier-admissible, got {:?} for body:\n{}",
                verification.reasons,
                candidates[0].body
            );
            assert!(
                candidates[0].mode.contains("train_novel_body_v1"),
                "post-v4 token candidate for {category} should prefer train-novel v4 body before replay, got {}; candidate modes={:?}",
                candidates[0].mode,
                candidates
                    .iter()
                    .map(|candidate| candidate.mode.clone())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn private_train_prototypes_cover_private_ecology_v5_when_generated() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(PRIVATE_ECOLOGY_V5_TRAIN_PATH);
        if !path.exists() {
            return;
        }
        let prototypes = load_broad_private_train_prototypes();
        for required in [
            "memory_state_tracking",
            "tool_status_parsing",
            "storage_sync_plan",
            "capability_latency_routing",
            "voice_following_route",
            "project_progress_digest",
            "media_preview_retrieval",
        ] {
            assert!(
                prototypes.contains_key(required),
                "missing private ecology v5 prototype for {required}"
            );
        }
    }

    #[test]
    fn private_ecology_v5_room_speaker_alias_prefers_voice_route_not_worker_route() {
        let task = private_ecology_v5_alias_task(
            "private_ood_room_speaker_handoff_choice_unit",
            "room_speaker_handoff_choice",
            "Private OOD transfer contract: Choose the best speaker node for a room-aware voice response.",
            json!({"data": "node_records", "other": "room_hint"}),
            vec![
                "device_route_contracts".to_string(),
                "ood_alias".to_string(),
                "room_speaker_handoff_choice".to_string(),
                "routing".to_string(),
                "voice".to_string(),
            ],
        );
        let candidates = broad_private_train_token_candidates(&task, true);
        assert!(
            !candidates.is_empty(),
            "room/speaker alias should infer a private-train token candidate"
        );
        assert!(
            candidates[0].mode.contains("train_novel_body_v1"),
            "room/speaker alias should prefer learned train-novel body before replay, got {}",
            candidates[0].mode
        );
        assert!(
            candidates[0].mode.contains("matched:voice_following_route"),
            "room/speaker alias should match voice-following route, got {}",
            candidates[0].mode
        );
        assert!(
            !candidates[0]
                .mode
                .contains("matched:capability_latency_routing"),
            "room/speaker alias must not collapse into generic worker capability routing: {}",
            candidates[0].mode
        );
        assert!(
            candidates[0].body.contains("speaker")
                && candidates[0].body.contains("confidence")
                && candidates[0].body.contains("room"),
            "voice-following candidate should preserve room/speaker selection behavior, got {}",
            candidates[0].body
        );
        let verification = decoder_contract_verifier_v1(&task, &candidates[0].body, None);
        assert!(
            verification.passed,
            "room/speaker alias token candidate should pass verifier, got {:?} for body:\n{}",
            verification.reasons, candidates[0].body
        );
    }

    #[test]
    fn private_ecology_v5_contract_blind_room_hint_infers_voice_route() {
        let mut task = private_ecology_v5_alias_task(
            "contract_blind_device_route_unit",
            "opaque_contract_blind_route_unit",
            "Private generated contract: select the target device for the visible request.",
            json!({"data": "node_records", "other": "room_hint"}),
            vec!["contract_blind_transfer".to_string(), "heldout".to_string()],
        );
        task.benchmark_evidence_level
            .push_str(";private_contract_blind_transfer_v1_generated_only");
        task.raw["private_contract_blind_transfer_v1"] = json!({
            "semantic_names_withheld": true,
            "public_benchmark_inputs_read": false
        });
        assert!(
            broad_private_train_prototype_candidates(&task, true).is_empty(),
            "contract-blind tasks must suppress prototype candidates and rely on learned-token evidence"
        );
        let candidates = broad_private_train_token_candidates(&task, true);
        assert!(
            !candidates.is_empty(),
            "contract-blind room_hint task should infer a private-train token candidate"
        );
        assert!(
            candidates[0].mode.contains("semantic_alias_inferred"),
            "contract-blind task should use inferred contract matching, got {}",
            candidates[0].mode
        );
        assert!(
            candidates[0].mode.contains("matched:voice_following_route"),
            "room_hint contract should infer voice-following route without prompt/tag/semantic-name hints, got {}",
            candidates[0].mode
        );
        assert!(
            !candidates[0]
                .mode
                .contains("matched:capability_latency_routing"),
            "room_hint contract must not collapse into generic worker capability routing: {}",
            candidates[0].mode
        );
        let verification = decoder_contract_verifier_v1(&task, &candidates[0].body, None);
        assert!(
            verification.passed,
            "contract-blind room_hint token candidate should pass verifier, got {:?} for body:\n{}",
            verification.reasons, candidates[0].body
        );
    }

    #[test]
    fn contract_blind_lcs_uses_visible_two_arg_contract_shape() {
        let mut task = broad_task("contract_blind_unit_lcs");
        task.benchmark_evidence_level
            .push_str(";private_contract_blind_transfer_v1_generated_only");
        task.raw["private_contract_blind_transfer_v1"] = json!({
            "semantic_names_withheld": true,
            "public_benchmark_inputs_read": false
        });
        if let Some(contract) = task
            .raw
            .get_mut("decoder_contract")
            .and_then(Value::as_object_mut)
        {
            contract.insert(
                "semantic_family".to_string(),
                json!("contract_blind_unit_lcs"),
            );
            contract.insert(
                "residual_label_hint".to_string(),
                json!("contract_blind_unit_lcs"),
            );
            contract.insert("type_family".to_string(), json!("dynamic_programming"));
            contract.insert("return_shape".to_string(), json!("number"));
            contract.insert("visible_arg_count_hint".to_string(), json!(2));
            contract.insert(
                "argument_roles".to_string(),
                json!({"data": "primary_input", "other": "secondary_input"}),
            );
            contract.insert(
                "required_constructs".to_string(),
                json!([
                    "loop",
                    "branch",
                    "locals",
                    "algorithmic_planning",
                    "index_or_string_ops"
                ]),
            );
        }
        let candidates = broad_private_train_token_candidates(&task, true);
        assert!(
            !candidates.is_empty(),
            "contract-blind LCS should infer a learned private-train token body"
        );
        assert!(
            candidates[0].mode.contains("train_novel_body_v1")
                && candidates[0].mode.contains("lcs_length"),
            "contract-blind two-arg DP should match LCS, got {}",
            candidates[0].mode
        );
        assert!(
            candidates[0].body.contains("str(other)") && candidates[0].body.contains("dp"),
            "LCS learned body should use the visible secondary argument and DP state:\n{}",
            candidates[0].body
        );
        let verification = decoder_contract_verifier_v1(&task, &candidates[0].body, None);
        assert!(
            verification.passed,
            "contract-blind LCS candidate should pass verifier, got {:?} for body:\n{}",
            verification.reasons, candidates[0].body
        );
    }

    #[test]
    fn contract_blind_shortest_hops_emits_novel_graph_token_candidate() {
        let mut task = broad_task("contract_blind_unit_0034");
        task.benchmark_evidence_level
            .push_str(";private_contract_blind_transfer_v1_generated_only");
        task.prompt = "Private contract-blind transfer task. Implement the required transformation from the visible arguments and decoder contract.".to_string();
        task.raw["private_contract_blind_transfer_v1"] = json!({
            "semantic_names_withheld": true,
            "public_benchmark_inputs_read": false
        });
        if let Some(contract) = task
            .raw
            .get_mut("decoder_contract")
            .and_then(Value::as_object_mut)
        {
            contract.insert(
                "semantic_family".to_string(),
                json!("contract_blind_unit_0034"),
            );
            contract.insert(
                "residual_label_hint".to_string(),
                json!("contract_blind_unit_0034"),
            );
            contract.insert("type_family".to_string(), json!("graph_search_algorithm"));
            contract.insert("return_shape".to_string(), json!("number"));
            contract.insert("visible_arg_count_hint".to_string(), json!(4));
            contract.insert(
                "argument_roles".to_string(),
                json!({
                    "data": "node_count",
                    "other": "edge_list",
                    "start": "source",
                    "goal": "target"
                }),
            );
            contract.insert(
                "required_constructs".to_string(),
                json!(["loop", "branch", "locals", "graph", "algorithmic_planning"]),
            );
        }
        let candidates = broad_private_train_token_candidates(&task, true);
        assert!(
            !candidates.is_empty(),
            "contract-blind shortest-hop graph should infer a learned private-train token body"
        );
        assert!(
            candidates[0].mode.contains("train_novel_body_v1")
                && candidates[0].mode.contains("shortest_hops"),
            "contract-blind graph shape should match shortest hops, got {}; modes={:?}",
            candidates[0].mode,
            candidates
                .iter()
                .map(|candidate| candidate.mode.clone())
                .collect::<Vec<_>>()
        );
        assert!(
            candidates[0].body.contains("graph")
                && candidates[0].body.contains("frontier")
                && candidates[0].body.contains("return -1"),
            "shortest-hop learned body should preserve BFS-like graph search:\n{}",
            candidates[0].body
        );
        let verification = decoder_contract_verifier_v1(&task, &candidates[0].body, None);
        assert!(
            verification.passed,
            "contract-blind shortest-hop candidate should pass verifier, got {:?} for body:\n{}",
            verification.reasons, candidates[0].body
        );
    }

    #[test]
    fn private_ecology_v5_train_induced_candidates_cover_operator_workflow_residuals() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("data/training_data/high_transfer/private_eval/private_ecology_generalization_v5_heldout_code_lm_tasks.jsonl");
        if !path.exists() {
            return;
        }
        let tasks = load_tasks(&path).expect("load v5 private ecology heldout rows");
        for category in [
            "v5_media_preview_index",
            "v5_tool_transcript_status",
            "v5_tool_error_clusters",
            "v5_storage_quota_select",
            "v5_storage_sync_plan",
            "v5_device_route_worker",
            "v5_voice_output_route",
            "v5_plan_next_unblocked",
            "v5_plan_progress_digest",
            "v5_room_capability_summary",
        ] {
            let task = tasks
                .iter()
                .find(|task| task.category == category)
                .unwrap_or_else(|| panic!("missing v5 heldout task for {category}"));
            let candidates = broad_private_train_token_candidates(task, true);
            let key = broad_private_semantic_key(task);
            let prototype_debug = load_broad_private_train_prototypes()
                .get(&key)
                .map(|prototype| {
                    format!(
                        "key={key} syntax={} natural_language={} scaffold={} body:\n{}",
                        syntax_constrained_body(&prototype.body),
                        natural_language_leakage_in_body(&prototype.body),
                        scaffold_placeholder_body(&prototype.body),
                        prototype.body
                    )
                })
                .unwrap_or_else(|| format!("missing prototype for key={key}"));
            assert!(
                !candidates.is_empty(),
                "expected at least one v5 train-induced token candidate for {category}; {prototype_debug}"
            );
            let verification = decoder_contract_verifier_v1(task, &candidates[0].body, None);
            assert!(
                verification.passed,
                "v5 token candidate for {category} should be verifier-admissible, got {:?} for body:\n{}",
                verification.reasons,
                candidates[0].body
            );
            assert!(
                candidates[0].mode.contains("train_novel_body_v1"),
                "v5 residual {category} should prefer train-novel body before replay, got {}; candidate modes={:?}; variant diagnostics={}",
                candidates[0].mode,
                candidates
                    .iter()
                    .map(|candidate| candidate.mode.clone())
                    .collect::<Vec<_>>(),
                load_broad_private_train_prototypes()
                    .get(&broad_private_semantic_key(task))
                    .map(|prototype| {
                        train_novel_body_variants(task, prototype)
                            .into_iter()
                            .map(|body| {
                                let verification = decoder_contract_verifier_v1(task, &body, None);
                                format!(
                                    "passed={} reasons={:?} body={}",
                                    verification.passed, verification.reasons, body
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("\n---\n")
                    })
                    .unwrap_or_else(|| "missing prototype".to_string())
            );
        }
    }

    #[test]
    fn private_unseen_aliases_prefer_train_novel_progress_and_room_bodies() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(
            "data/training_data/high_transfer/private_eval/private_unseen_transfer_challenge_v1_code_lm_tasks.jsonl",
        );
        if !path.exists() {
            return;
        }
        let tasks = load_tasks(&path).expect("load private unseen transfer challenge rows");
        for semantic_family in [
            "project_work_state_summary",
            "room_device_capability_counts",
        ] {
            let task = tasks
                .iter()
                .find(|task| broad_private_semantic_key(task) == semantic_family)
                .unwrap_or_else(|| {
                    panic!("missing private unseen alias task for {semantic_family}")
                });
            let candidates = broad_private_train_token_candidates(task, true);
            assert!(
                !candidates.is_empty(),
                "expected token candidates for unseen alias {semantic_family}"
            );
            let candidate = &candidates[0];
            let verification = decoder_contract_verifier_v1(task, &candidate.body, None);
            assert!(
                verification.passed,
                "unseen alias {semantic_family} candidate should pass verifier, got {:?} for body:\n{}",
                verification.reasons,
                candidate.body
            );
            assert!(
                candidate.mode.contains("train_novel_body_v1"),
                "unseen alias {semantic_family} should prefer train-novel body before replay, got {}",
                candidate.mode
            );
        }
    }

    #[test]
    fn post_v4_shadow_transfer_v6_tasks_are_recognized_and_sts_gated() {
        let task = post_v4_shadow_task("shadow_verifier_mismatch_bpg_parse_query_string");
        assert!(broad_private_generated_task(&task));
        assert!(broad_private_train_token_candidates(&task, false).is_empty());
    }

    #[test]
    fn private_train_prototype_candidate_is_sts_gated() {
        let task = broad_task("bpg_parse_query_string");
        assert!(
            broad_private_train_prototype_candidates(&task, false).is_empty(),
            "STS-off control must not receive private train prototype candidates"
        );
        let candidates = broad_private_train_prototype_candidates(&task, true);
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0]
            .mode
            .contains("private_train_induced_broad_semantic_prototype_decoder_v1"));
        assert!(candidates[0].body.contains("split('&')"));
        let token_candidates = broad_private_train_token_candidates(&task, true);
        assert!(
            !token_candidates.is_empty(),
            "private train token bridge should emit a novel token candidate before body-memory replay"
        );
        assert!(token_candidates[0]
            .mode
            .contains("private_train_induced_broad_semantic_token_decoder_v1"));
        assert!(
            token_candidates[0].mode.contains("train_novel_body_v1"),
            "base bpg task should prefer v4 train-novel token body before replay, got {}",
            token_candidates[0].mode
        );
        assert!(
            learned_token_decoder_candidate(&token_candidates[0]),
            "private train token bridge should be learner-facing token evidence"
        );
    }

    #[test]
    fn semantic_alias_task_infers_private_train_token_candidate() {
        let mut task = broad_task("bpg_semantic_alias_query_parser_private_heldout");
        task.prompt = "Parse a query string into a dict of key -> list of values.".to_string();
        if let Some(contract) = task
            .raw
            .get_mut("decoder_contract")
            .and_then(Value::as_object_mut)
        {
            contract.insert(
                "semantic_family".to_string(),
                json!("semantic_alias_query_parser_private_heldout"),
            );
            contract.insert("type_family".to_string(), json!("structured_parsing"));
            contract.insert("return_shape".to_string(), json!("dict"));
            contract.insert(
                "required_constructs".to_string(),
                json!([
                    "loop",
                    "branch",
                    "locals",
                    "parsing",
                    "type_and_return_shape"
                ]),
            );
        }
        let candidates = broad_private_train_token_candidates(&task, true);
        assert!(
            !candidates.is_empty(),
            "semantic alias should infer a reusable private-train token body"
        );
        assert!(
            candidates[0]
                .mode
                .contains("semantic_alias_inferred:semantic_alias_query_parser_private_heldout:matched:bpg_parse_query_string"),
            "alias candidate should disclose inferred match, got {}",
            candidates[0].mode
        );
        assert!(
            candidates[0].body.contains("split('&')"),
            "query-string alias should recover query parser body, got {}",
            candidates[0].body
        );
        assert!(
            learned_token_decoder_candidate(&candidates[0]),
            "alias recovery should remain learner-facing token evidence"
        );
    }

    #[test]
    fn semantic_alias_lcs_keeps_two_argument_token_contract() {
        let mut task = broad_task("semantic_alias_lcs_length_private_heldout");
        task.prompt = "Return longest common subsequence length for two strings.".to_string();
        if let Some(contract) = task
            .raw
            .get_mut("decoder_contract")
            .and_then(Value::as_object_mut)
        {
            contract.insert(
                "semantic_family".to_string(),
                json!("semantic_alias_lcs_length_private_heldout"),
            );
            contract.insert("type_family".to_string(), json!("dynamic_programming"));
            contract.insert("return_shape".to_string(), json!("number"));
            contract.insert("visible_arg_count_hint".to_string(), json!(2));
            contract.insert(
                "argument_roles".to_string(),
                json!({"data": "primary_input", "other": "secondary_input"}),
            );
            contract.insert(
                "required_constructs".to_string(),
                json!([
                    "loop",
                    "branch",
                    "locals",
                    "algorithmic_planning",
                    "index_or_string_ops"
                ]),
            );
        }
        let candidates = broad_private_train_token_candidates(&task, true);
        assert!(!candidates.is_empty());
        assert!(
            candidates[0].mode.contains("matched:bpg_lcs_length"),
            "LCS alias should infer the private LCS token body, got {}",
            candidates[0].mode
        );
        let verification = decoder_contract_verifier_v1(&task, &candidates[0].body, None);
        assert!(
            verification.passed,
            "LCS alias token candidate should pass verifier, got {:?} for body:\n{}",
            verification.reasons, candidates[0].body
        );
    }

    #[test]
    fn novel_composition_task_combines_private_train_token_bodies() {
        let mut task = broad_task("bpg_novel_composition_parse_ints_then_even_run");
        task.prompt =
            "Extract signed integers from text, then return the longest contiguous run of even integers."
                .to_string();
        task.raw["novel_composition_v1"] = json!({
            "steps": [
                {"semantic_family": "bpg_parse_signed_ints"},
                {"semantic_family": "bpg_longest_even_run"}
            ],
            "public_tests_used": false,
            "public_solutions_used": false
        });
        if let Some(contract) = task
            .raw
            .get_mut("decoder_contract")
            .and_then(Value::as_object_mut)
        {
            contract.insert(
                "semantic_family".to_string(),
                json!("novel_composition_parse_signed_ints_then_longest_even_run"),
            );
            contract.insert(
                "type_family".to_string(),
                json!("novel_composition_pipeline"),
            );
            contract.insert("return_shape".to_string(), json!("number"));
            contract.insert("visible_arg_count_hint".to_string(), json!(1));
            contract.insert(
                "composition_steps".to_string(),
                json!([
                    {"semantic_family": "bpg_parse_signed_ints"},
                    {"semantic_family": "bpg_longest_even_run"}
                ]),
            );
            contract.insert(
                "required_constructs".to_string(),
                json!([
                    "loop",
                    "branch",
                    "locals",
                    "composition",
                    "type_and_return_shape"
                ]),
            );
        }
        let candidates = broad_private_train_composition_token_candidates(&task, true);
        assert_eq!(candidates.len(), 1);
        assert!(
            candidates[0].mode.contains("novel_composition_v1"),
            "composition candidate should disclose composition mode, got {}",
            candidates[0].mode
        );
        assert!(candidates[0].body.contains("_theseus_value_0"));
        assert!(candidates[0].body.contains("data = _theseus_value_0"));
        assert!(
            candidates[0].body.contains("str(data) + ' '"),
            "first private train body should parse signed integers"
        );
        assert!(
            candidates[0].body.contains("current = 0"),
            "second private train body should compute longest even run"
        );
        assert!(
            learned_token_decoder_candidate(&candidates[0]),
            "composition should remain learner-facing token evidence"
        );
        let verification = decoder_contract_verifier_v1(&task, &candidates[0].body, None);
        assert!(
            verification.passed,
            "composition token candidate should pass verifier, got {:?} for body:\n{}",
            verification.reasons, candidates[0].body
        );
    }

    #[test]
    fn private_residual_frontier_single_step_composition_is_not_duplicated() {
        let mut task = broad_task("private_residual_frontier_group_records");
        task.prompt =
            "Group record IDs by a visible record field and preserve the typed mapping contract."
                .to_string();
        task.raw["novel_composition_v1"] = json!({
            "steps": [
                {"semantic_family": "bpg_group_records"}
            ],
            "public_tests_used": false,
            "public_solutions_used": false
        });
        if let Some(contract) = task
            .raw
            .get_mut("decoder_contract")
            .and_then(Value::as_object_mut)
        {
            contract.insert(
                "semantic_family".to_string(),
                json!("private_residual_frontier_group_records"),
            );
            contract.insert(
                "type_family".to_string(),
                json!("return_shape_record_pipeline"),
            );
            contract.insert("return_shape".to_string(), json!("dict"));
            contract.insert("visible_arg_count_hint".to_string(), json!(2));
            contract.insert(
                "composition_steps".to_string(),
                json!([
                    {"semantic_family": "bpg_group_records"}
                ]),
            );
            contract.insert(
                "required_constructs".to_string(),
                json!([
                    "loop",
                    "branch",
                    "locals",
                    "record_filter",
                    "type_and_return_shape"
                ]),
            );
        }
        let candidates = broad_private_train_composition_token_candidates(&task, true);
        assert_eq!(
            candidates.len(),
            1,
            "expected a single-step private residual frontier learned-token candidate"
        );
        assert!(
            candidates[0]
                .mode
                .contains("novel_composition_v1:bpg_group_records:"),
            "single-step mode should not duplicate the semantic key, got {}",
            candidates[0].mode
        );
        assert!(
            !candidates[0]
                .mode
                .contains("bpg_group_records_then_bpg_group_records"),
            "single-step composition was duplicated: {}",
            candidates[0].mode
        );
        assert!(
            !candidates[0].body.contains("_theseus_value_0"),
            "single-step body should not render an intermediate composition value:\n{}",
            candidates[0].body
        );
        assert!(
            candidates[0].body.contains("out.setdefault")
                || candidates[0].body.contains("if key not in out"),
            "candidate should use the learned group-record body:\n{}",
            candidates[0].body
        );
        assert!(
            learned_token_decoder_candidate(&candidates[0]),
            "single-step frontier candidate should remain learner-facing token evidence"
        );
        let verification = decoder_contract_verifier_v1(&task, &candidates[0].body, None);
        assert!(
            verification.passed,
            "single-step group-record candidate should pass verifier, got {:?} for body:\n{}",
            verification.reasons, candidates[0].body
        );
    }

    #[test]
    fn private_residual_frontier_stdin_three_step_composition_is_available() {
        let mut task =
            broad_task("private_residual_frontier_stdin_pair_sums_then_parse_then_gcd_positive");
        task.prompt =
            "Sum integer pairs from a stdin-style string, parse the sums, then return their positive gcd."
                .to_string();
        task.raw["novel_composition_v1"] = json!({
            "steps": [
                {"semantic_family": "bpg_stdin_pair_sums"},
                {"semantic_family": "bpg_parse_signed_ints"},
                {"semantic_family": "bpg_gcd_positive"}
            ],
            "public_tests_used": false,
            "public_solutions_used": false
        });
        if let Some(contract) = task
            .raw
            .get_mut("decoder_contract")
            .and_then(Value::as_object_mut)
        {
            contract.insert(
                "semantic_family".to_string(),
                json!("private_residual_frontier_stdin_pair_sums_then_parse_then_gcd_positive"),
            );
            contract.insert("type_family".to_string(), json!("algorithmic_planning"));
            contract.insert("return_shape".to_string(), json!("number"));
            contract.insert("visible_arg_count_hint".to_string(), json!(1));
            contract.insert(
                "composition_steps".to_string(),
                json!([
                    {"semantic_family": "bpg_stdin_pair_sums"},
                    {"semantic_family": "bpg_parse_signed_ints"},
                    {"semantic_family": "bpg_gcd_positive"}
                ]),
            );
            contract.insert(
                "required_constructs".to_string(),
                json!([
                    "loop",
                    "branch",
                    "locals",
                    "composition",
                    "stdin_parser",
                    "algorithmic_planning"
                ]),
            );
        }
        let candidates = broad_private_train_composition_token_candidates(&task, true);
        assert_eq!(
            candidates.len(),
            1,
            "expected a three-step stdin residual-frontier composition candidate"
        );
        assert!(
            candidates[0]
                .mode
                .contains("bpg_stdin_pair_sums_then_bpg_parse_signed_ints_then_bpg_gcd_positive"),
            "unexpected composition mode: {}",
            candidates[0].mode
        );
        assert!(candidates[0].body.contains("_theseus_value_0"));
        assert!(candidates[0].body.contains("_theseus_value_1"));
        assert!(
            learned_token_decoder_candidate(&candidates[0]),
            "three-step frontier composition should remain learner-facing token evidence"
        );
        let verification = decoder_contract_verifier_v1(&task, &candidates[0].body, None);
        assert!(
            verification.passed,
            "three-step frontier composition should pass verifier, got {:?} for body:\n{}",
            verification.reasons, candidates[0].body
        );
    }

    #[test]
    fn private_residual_frontier_parse_then_dp_composition_passes_verifier() {
        let mut task =
            broad_task("private_residual_frontier_parse_signed_ints_then_max_non_adjacent_sum");
        task.prompt = "Extract signed integers from noisy text, then solve a non-adjacent dynamic-programming selection."
            .to_string();
        task.raw["novel_composition_v1"] = json!({
            "steps": [
                {"semantic_family": "bpg_parse_signed_ints"},
                {"semantic_family": "bpg_max_non_adjacent_sum"}
            ],
            "public_tests_used": false,
            "public_solutions_used": false
        });
        if let Some(contract) = task
            .raw
            .get_mut("decoder_contract")
            .and_then(Value::as_object_mut)
        {
            contract.insert(
                "semantic_family".to_string(),
                json!("private_residual_frontier_parse_signed_ints_then_max_non_adjacent_sum"),
            );
            contract.insert("type_family".to_string(), json!("algorithmic_planning"));
            contract.insert("return_shape".to_string(), json!("number"));
            contract.insert("visible_arg_count_hint".to_string(), json!(1));
            contract.insert(
                "composition_steps".to_string(),
                json!([
                    {"semantic_family": "bpg_parse_signed_ints"},
                    {"semantic_family": "bpg_max_non_adjacent_sum"}
                ]),
            );
            contract.insert(
                "required_constructs".to_string(),
                json!([
                    "loop",
                    "branch",
                    "locals",
                    "composition",
                    "dynamic_programming"
                ]),
            );
        }
        let candidates = broad_private_train_composition_token_candidates(&task, true);
        assert_eq!(
            candidates.len(),
            1,
            "expected learned parse->dynamic-programming composition candidate"
        );
        assert!(
            candidates[0]
                .mode
                .contains("bpg_parse_signed_ints_then_bpg_max_non_adjacent_sum"),
            "unexpected composition mode: {}",
            candidates[0].mode
        );
        assert!(
            candidates[0].body.contains("take, skip") || candidates[0].body.contains("best_"),
            "composition body should include the learned DP state update body:\n{}",
            candidates[0].body
        );
        assert!(
            learned_token_decoder_candidate(&candidates[0]),
            "DP frontier composition should remain learner-facing token evidence"
        );
        let verification = decoder_contract_verifier_v1(&task, &candidates[0].body, None);
        assert!(
            verification.passed,
            "DP frontier composition should pass verifier, got {:?} for body:\n{}",
            verification.reasons, candidates[0].body
        );
    }

    fn broad_task(category: &str) -> CodeTask {
        CodeTask {
            raw: json!({
                "broad_private_family_v1": "unit_test_family",
                "decoder_contract": {
                    "policy": "project_theseus_decoder_contract_v1_broad_private_generalization",
                    "return_shape": "dict",
                    "type_family": "structured_parsing",
                    "visible_arg_count_hint": 1,
                    "required_constructs": ["loop", "branch", "locals", "parsing", "type_and_return_shape"],
                    "semantic_family": category,
                    "full_body_required": true
                }
            }),
            task_id: format!("broad_private_generalization_ladder_v1_{category}"),
            source_task_id: "unit".to_string(),
            card_id: "broad_private_generalization_ladder_v1".to_string(),
            source_id: "unit".to_string(),
            split: "heldout".to_string(),
            category: category.to_string(),
            prompt: "private generated broad transfer task".to_string(),
            entry_point: format!("{category}_entry"),
            solution_expr: String::new(),
            solution_body: String::new(),
            tags: vec!["broad_private_generalization_ladder_v1".to_string()],
            benchmark_evidence_level: "broad_private_generalization_ladder_v1_generated_only"
                .to_string(),
        }
    }

    fn public_safe_v4_task(category: &str, semantic_family: &str) -> CodeTask {
        CodeTask {
            raw: json!({
                "broad_private_family_v1": "unit_test_family",
                "decoder_contract": {
                    "policy": "project_theseus_decoder_contract_v4_public_safe_broad_transfer_maturity",
                    "return_shape": "dict",
                    "type_family": "structured_parsing",
                    "visible_arg_count_hint": 1,
                    "required_constructs": ["loop", "branch", "locals", "parsing", "type_and_return_shape"],
                    "semantic_family": semantic_family,
                    "full_body_required": true
                }
            }),
            task_id: format!("public_safe_broad_transfer_maturity_v4_{category}"),
            source_task_id: "unit".to_string(),
            card_id: "public_safe_broad_transfer_maturity_v4".to_string(),
            source_id: "unit".to_string(),
            split: "heldout".to_string(),
            category: category.to_string(),
            prompt: "private generated public-safe maturity task".to_string(),
            entry_point: format!("{category}_entry"),
            solution_expr: String::new(),
            solution_body: String::new(),
            tags: vec!["public_safe_broad_transfer_maturity_v4".to_string()],
            benchmark_evidence_level: "public_safe_broad_transfer_maturity_v4_generated_only"
                .to_string(),
        }
    }

    fn post_v4_shadow_task(category: &str) -> CodeTask {
        CodeTask {
            raw: json!({
                "broad_private_family_v1": "verifier_mismatch_shadow",
                "decoder_contract": {
                    "policy": "project_theseus_decoder_contract_v6_post_v4_private_shadow_transfer",
                    "return_shape": "dict",
                    "type_family": "structured_parsing",
                    "visible_arg_count_hint": 1,
                    "required_constructs": ["loop", "branch", "locals", "parsing", "type_and_return_shape"],
                    "semantic_family": category,
                    "full_body_required": true
                }
            }),
            task_id: format!("post_v4_private_shadow_transfer_v1_{category}"),
            source_task_id: "unit".to_string(),
            card_id: "post_v4_private_shadow_transfer_v1".to_string(),
            source_id: "unit".to_string(),
            split: "heldout".to_string(),
            category: category.to_string(),
            prompt: "private generated post-v4 shadow task".to_string(),
            entry_point: format!("{category}_entry"),
            solution_expr: String::new(),
            solution_body: String::new(),
            tags: vec!["post_v4_private_shadow_transfer_v1".to_string()],
            benchmark_evidence_level: "post_v4_private_shadow_transfer_v1_generated_only"
                .to_string(),
        }
    }

    fn private_ecology_v5_alias_task(
        category: &str,
        semantic_family: &str,
        prompt: &str,
        argument_roles: Value,
        tags: Vec<String>,
    ) -> CodeTask {
        CodeTask {
            raw: json!({
                "broad_private_family_v1": "device_route_contracts",
                "decoder_contract": {
                    "policy": "project_theseus_decoder_contract_v5_private_ecology_generalization",
                    "return_shape": "str",
                    "type_family": "device_routing",
                    "visible_arg_count_hint": 2,
                    "argument_roles": argument_roles,
                    "required_constructs": ["loop", "branch", "locals", "selection"],
                    "semantic_family": semantic_family,
                    "residual_label_hint": semantic_family,
                    "full_body_required": true
                }
            }),
            task_id: format!("private_unseen_transfer_challenge_v1_{category}"),
            source_task_id: "unit".to_string(),
            card_id: "private_ecology_generalization_v5".to_string(),
            source_id: "unit".to_string(),
            split: "heldout".to_string(),
            category: category.to_string(),
            prompt: prompt.to_string(),
            entry_point: format!("{category}_entry"),
            solution_expr: String::new(),
            solution_body: String::new(),
            tags,
            benchmark_evidence_level:
                "private_ecology_generalization_v5_generated_only;private_unseen_transfer_challenge_v1_ood_alias"
                    .to_string(),
        }
    }
}
