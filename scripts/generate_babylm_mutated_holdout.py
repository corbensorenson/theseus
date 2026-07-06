"""Generate local mutated BabyLM/BLIMP-style holdouts.

The factory is deliberately local-only: it reads existing JSONL minimal-pair
snapshots and residual reports, then creates fresh deterministic minimal pairs
for the highest-pressure grammar families. No model/provider inference is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCES = [
    "data/babylm_blimp_filtered_train.jsonl",
    "data/babylm_blimp_filtered_eval.jsonl",
    "data/public_blimp_train.jsonl",
    "data/public_blimp_eval.jsonl",
]

IRREGULAR_NOUNS = [
    ("child", "children"),
    ("man", "men"),
    ("woman", "women"),
    ("person", "people"),
    ("mouse", "mice"),
    ("goose", "geese"),
    ("criterion", "criteria"),
    ("phenomenon", "phenomena"),
    ("stimulus", "stimuli"),
    ("analysis", "analyses"),
    ("thesis", "theses"),
    ("diagnosis", "diagnoses"),
]

ADJECTIVES = [
    "blue",
    "green",
    "old",
    "young",
    "quiet",
    "careful",
    "local",
    "bright",
    "formal",
    "silver",
]

PLACES = [
    "near the window",
    "beside the library",
    "behind the stage",
    "in the hallway",
    "by the archive",
]

FEMALE_NAMES = ["Alice", "Carla", "Diane", "Eva", "Laura", "Natalie"]
MALE_NAMES = ["Bruce", "Carl", "Eric", "James", "Mark", "Patrick"]
PLURAL_GROUPS = ["children", "students", "teachers", "patients", "drivers", "dancers"]
SINGULAR_PEOPLE = ["child", "student", "teacher", "patient", "driver", "dancer"]
INANIMATE_NOUNS = ["window", "screen", "ladder", "projector", "book", "bicycle"]
OBJECTS = ["package", "sketch", "report", "box", "letter", "vase"]
TRANSITIVE_VERBS = ["carried", "admired", "moved", "praised", "criticized", "inspected"]
INTRANSITIVE_VERBS = ["slept", "arrived", "laughed", "waited", "skated", "swam"]

PRESERVING_REPLACEMENTS = {
    "alice": ["carla", "diane", "eva", "laura"],
    "bruce": ["carl", "eric", "james", "mark"],
    "child": ["person", "woman", "man"],
    "children": ["people", "women", "men"],
    "student": ["teacher", "patient", "driver"],
    "students": ["teachers", "patients", "drivers"],
    "girl": ["woman", "actress", "waitress"],
    "girls": ["women", "actresses", "waitresses"],
    "boy": ["man", "actor", "waiter"],
    "boys": ["men", "actors", "waiters"],
    "window": ["screen", "projector", "book"],
    "windows": ["screens", "projectors", "books"],
    "blue": ["green", "silver", "bright"],
    "old": ["young", "careful", "quiet"],
    "near": ["beside", "behind"],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="JSONL files whose exact minimal pairs must be excluded from the generated holdout.",
    )
    parser.add_argument(
        "--exclude-sentences",
        action="store_true",
        help="Also exclude generated rows if either sentence appears in sources or excluded files.",
    )
    parser.add_argument(
        "--residual-analysis", default="reports/babylm_residual_analysis.json"
    )
    parser.add_argument("--out", default="data/babylm_mutated_holdout_seed31.jsonl")
    parser.add_argument(
        "--report-out", default="reports/babylm_mutated_holdout_seed31_factory.json"
    )
    parser.add_argument("--count", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--target-rule", action="append", default=[])
    parser.add_argument("--target-term", action="append", default=[])
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sources = args.source or DEFAULT_SOURCES
    residual = read_json(Path(args.residual_analysis), {})
    target_rules = normalize_targets(
        args.target_rule or residual.get("recommendation", {}).get("target_rules", [])
    )
    target_terms = normalize_targets(
        args.target_term or residual.get("recommendation", {}).get("target_terms", [])
    )
    source_rows = read_source_rows([Path(path) for path in sources])
    exclude_rows = read_source_rows([Path(path) for path in args.exclude])
    target_rows = filter_target_rows(source_rows, target_rules, target_terms)

    generated: list[dict[str, Any]] = []
    generated_pairs: set[str] = set()
    generated_sentences: set[str] = set()
    source_pairs = {
        pair_key(row.get("sentence_good", ""), row.get("sentence_bad", ""))
        for row in source_rows + exclude_rows
    }
    source_sentences = {
        sentence_key(sentence)
        for row in source_rows + exclude_rows
        for sentence in (row.get("sentence_good", ""), row.get("sentence_bad", ""))
        if str(sentence).strip()
    }
    template_rows = generate_template_rows(rng, args.count * 3, target_rules, target_terms)
    for row in template_rows:
        append_if_new(
            generated,
            row,
            source_pairs,
            source_sentences if args.exclude_sentences else set(),
            generated_pairs,
            generated_sentences,
            args.count,
        )
        if len(generated) >= args.count:
            break

    mutation_attempts = 0
    while len(generated) < args.count and target_rows and mutation_attempts < args.count * 20:
        mutation_attempts += 1
        row = rng.choice(target_rows)
        mutated = mutate_source_row(row, rng)
        if mutated is None:
            continue
        append_if_new(
            generated,
            mutated,
            source_pairs,
            source_sentences if args.exclude_sentences else set(),
            generated_pairs,
            generated_sentences,
            args.count,
        )

    if len(generated) < args.count:
        fallback_rows = generate_template_rows(rng, args.count * 2, [], [])
        for row in fallback_rows:
            append_if_new(
                generated,
                row,
                source_pairs,
                source_sentences if args.exclude_sentences else set(),
                generated_pairs,
                generated_sentences,
                args.count,
            )
            if len(generated) >= args.count:
                break

    rng.shuffle(generated)
    for idx, row in enumerate(generated):
        row["pair_id"] = idx
        row["holdout_id"] = f"babylm_mutated_{args.seed}_{idx:05}"
        row["seed"] = args.seed

    write_jsonl(Path(args.out), generated)
    report = build_report(
        generated=generated,
        sources=sources,
        excludes=args.exclude,
        sentence_exclusion=args.exclude_sentences,
        residual_analysis=args.residual_analysis,
        target_rules=target_rules,
        target_terms=target_terms,
        out=args.out,
        seed=args.seed,
    )
    write_json(Path(args.report_out), report)
    print(json.dumps(report, indent=2))
    return 0


def normalize_targets(values: Iterable[str]) -> list[str]:
    targets: list[str] = []
    for value in values:
        for part in str(value).split(","):
            normalized = part.strip()
            if normalized and normalized not in targets:
                targets.append(normalized)
    return targets


def read_source_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("sentence_good") and row.get("sentence_bad"):
                row = dict(row)
                row.setdefault("source_path", str(path))
                rows.append(row)
    return rows


def filter_target_rows(
    rows: list[dict[str, Any]], target_rules: list[str], target_terms: list[str]
) -> list[dict[str, Any]]:
    if not target_rules and not target_terms:
        return rows
    rule_set = set(target_rules)
    term_set = set(target_terms)
    filtered = [
        row
        for row in rows
        if row.get("rule") in rule_set
        or row.get("UID") in rule_set
        or row.get("linguistics_term") in term_set
    ]
    return filtered or rows


def append_if_new(
    rows: list[dict[str, Any]],
    candidate: dict[str, Any],
    source_pairs: set[str],
    source_sentences: set[str],
    generated_pairs: set[str],
    generated_sentences: set[str],
    limit: int,
) -> None:
    if len(rows) >= limit:
        return
    good = clean_sentence(str(candidate.get("sentence_good", "")))
    bad = clean_sentence(str(candidate.get("sentence_bad", "")))
    if not good or not bad or good == bad:
        return
    candidate["sentence_good"] = good
    candidate["sentence_bad"] = bad
    key = pair_key(good, bad)
    if key in source_pairs:
        return
    if source_sentences and (
        sentence_key(good) in source_sentences or sentence_key(bad) in source_sentences
    ):
        return
    if key in generated_pairs:
        return
    if source_sentences:
        candidate_sentences = {sentence_key(good), sentence_key(bad)}
        if candidate_sentences & generated_sentences:
            return
    candidate.setdefault("source", "symliquid_babylm_mutated_holdout_factory")
    candidate.setdefault("mutation_policy", "local_only_no_external_inference")
    rows.append(candidate)
    generated_pairs.add(key)
    generated_sentences.add(sentence_key(good))
    generated_sentences.add(sentence_key(bad))


def clean_sentence(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence.strip())
    if sentence and sentence[-1] not in ".?!":
        sentence += "."
    return sentence


def pair_key(good: str, bad: str) -> str:
    return hashlib.sha256(
        f"{good.strip().lower()}\n{bad.strip().lower()}".encode("utf-8")
    ).hexdigest()


def sentence_key(sentence: str) -> str:
    normalized = re.sub(r"\s+", " ", str(sentence).strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def generate_template_rows(
    rng: random.Random,
    limit: int,
    target_rules: list[str],
    target_terms: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    generators = [
        template_determiner_irregular,
        template_subject_verb_irregular,
        template_distractor_agreement,
        template_binding_domain,
        template_principle_a_reconstruction,
        template_ellipsis,
        template_wh_gap,
        template_argument_structure,
        template_animate_subject,
    ]
    while len(rows) < limit:
        generator = rng.choice(generators)
        row = generator(rng)
        if target_rules or target_terms:
            target_hit = row["rule"] in target_rules or row["linguistics_term"] in target_terms
            if not target_hit and rng.random() < 0.75:
                continue
        rows.append(row)
    return rows


def template_determiner_irregular(rng: random.Random) -> dict[str, Any]:
    singular, plural = rng.choice(IRREGULAR_NOUNS)
    adjective = rng.choice(ADJECTIVES)
    place = rng.choice(PLACES)
    plural_case = rng.random() < 0.5
    if plural_case:
        good = f"Those {adjective} {plural} {place} are quiet"
        bad = f"That {adjective} {plural} {place} are quiet"
        rule = "determiner_noun_agreement_with_adj_irregular_2"
    else:
        good = f"This {adjective} {singular} {place} is quiet"
        bad = f"These {adjective} {singular} {place} is quiet"
        rule = "determiner_noun_agreement_with_adj_irregular_1"
    return row(good, bad, rule, "morphology", "determiner_noun_agreement", "template")


def template_subject_verb_irregular(rng: random.Random) -> dict[str, Any]:
    singular, plural = rng.choice(IRREGULAR_NOUNS)
    adjective = rng.choice(ADJECTIVES)
    place = rng.choice(PLACES)
    plural_case = rng.random() < 0.65
    if plural_case:
        good = f"The {adjective} {plural} {place} are ready"
        bad = f"The {adjective} {plural} {place} is ready"
        rule = "irregular_plural_subject_verb_agreement_1"
    else:
        good = f"The {adjective} {singular} {place} is ready"
        bad = f"The {adjective} {singular} {place} are ready"
        rule = "irregular_plural_subject_verb_agreement_2"
    return row(good, bad, rule, "morphology", "subject_verb_agreement", "template")


def template_distractor_agreement(rng: random.Random) -> dict[str, Any]:
    singular, plural = rng.choice(IRREGULAR_NOUNS)
    distractor = rng.choice([noun for pair in IRREGULAR_NOUNS for noun in pair])
    if rng.random() < 0.5:
        good = f"The {singular} near the {distractor} is calm"
        bad = f"The {singular} near the {distractor} are calm"
    else:
        good = f"The {plural} near the {distractor} are calm"
        bad = f"The {plural} near the {distractor} is calm"
    return row(
        good,
        bad,
        "distractor_agreement_relative_clause",
        "morphology",
        "subject_verb_agreement",
        "template",
    )


def template_binding_domain(rng: random.Random) -> dict[str, Any]:
    if rng.random() < 0.5:
        matrix = rng.choice(MALE_NAMES)
        local = rng.choice(FEMALE_NAMES)
        good = f"{matrix} said that {local} praised herself"
        bad = f"{matrix} said that {local} praised himself"
    else:
        matrix = rng.choice(FEMALE_NAMES)
        local = rng.choice(PLURAL_GROUPS)
        good = f"{matrix} said that the {local} blamed themselves"
        bad = f"{matrix} said that the {local} blamed herself"
    return row(good, bad, "principle_A_domain_1", "syntax_semantics", "binding", "template")


def template_principle_a_reconstruction(rng: random.Random) -> dict[str, Any]:
    if rng.random() < 0.5:
        name = rng.choice(FEMALE_NAMES)
        good = f"It's herself that {name} criticized"
        bad = f"It's himself that {name} criticized"
    else:
        name = rng.choice(MALE_NAMES)
        good = f"It's himself that {name} criticized"
        bad = f"It's herself that {name} criticized"
    return row(
        good,
        bad,
        "principle_A_reconstruction",
        "syntax_semantics",
        "binding",
        "template",
    )


def template_ellipsis(rng: random.Random) -> dict[str, Any]:
    group_a = rng.choice(PLURAL_GROUPS)
    group_b = rng.choice([g for g in PLURAL_GROUPS if g != group_a])
    adjective_a, adjective_b = rng.sample(ADJECTIVES, 2)
    obj = rng.choice(["pencils", "reports", "sketches", "books", "vases"])
    count_a, count_b = rng.choice([("three", "two"), ("four", "several"), ("many", "few")])
    good = (
        f"The {group_a} bought {count_a} {adjective_a} {obj}, "
        f"and the {group_b} bought {count_b} {adjective_b} ones"
    )
    bad = (
        f"The {group_a} bought {count_a} {adjective_a} {obj}, "
        f"and the {group_b} bought {count_b} {adjective_b}"
    )
    rule = rng.choice(["ellipsis_n_bar_1", "ellipsis_n_bar_2"])
    return row(good, bad, rule, "syntax", "ellipsis", "template")


def template_wh_gap(rng: random.Random) -> dict[str, Any]:
    subject = rng.choice(SINGULAR_PEOPLE)
    obj = rng.choice(SINGULAR_PEOPLE + OBJECTS)
    verb = rng.choice(["praised", "admired", "criticized", "inspected"])
    if rng.random() < 0.5:
        good = f"The reporter knew who the {subject} {verb}"
        bad = f"The reporter knew that the {subject} {verb}"
        rule = "wh_vs_that_with_gap"
    else:
        good = f"The reporter knew that the {subject} {verb} the {obj}"
        bad = f"The reporter knew who the {subject} {verb} the {obj}"
        rule = "wh_vs_that_no_gap"
    return row(good, bad, rule, "syntax", "filler_gap_dependency", "template")


def template_argument_structure(rng: random.Random) -> dict[str, Any]:
    person = rng.choice(SINGULAR_PEOPLE)
    obj = rng.choice(OBJECTS)
    if rng.random() < 0.5:
        verb = rng.choice(TRANSITIVE_VERBS)
        good = f"The {person} {verb} the {obj}"
        bad = f"The {person} {verb}"
        rule = "transitive"
    else:
        verb = rng.choice(INTRANSITIVE_VERBS)
        good = f"The {person} {verb}"
        bad = f"The {person} {verb} the {obj}"
        rule = "intransitive"
    return row(good, bad, rule, "syntax", "argument_structure", "template")


def template_animate_subject(rng: random.Random) -> dict[str, Any]:
    animate = rng.choice(SINGULAR_PEOPLE)
    inanimate = rng.choice(INANIMATE_NOUNS)
    verb = rng.choice(["laughed", "swam", "skated", "waited"])
    good = f"The {animate} {verb} near the stage"
    bad = f"The {inanimate} {verb} near the stage"
    return row(good, bad, "animate_subject_trans", "syntax_semantics", "animacy", "template")


def row(
    good: str,
    bad: str,
    rule: str,
    field: str,
    term: str,
    mutation_kind: str,
) -> dict[str, Any]:
    return {
        "sentence_good": good,
        "sentence_bad": bad,
        "rule": rule,
        "field": field,
        "linguistics_term": term,
        "source": "symliquid_babylm_mutated_holdout_factory",
        "mutation_kind": mutation_kind,
    }


def mutate_source_row(row_in: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    good = str(row_in.get("sentence_good", ""))
    bad = str(row_in.get("sentence_bad", ""))
    mutated_good = mutate_sentence(good, rng)
    mutated_bad = mutate_sentence(bad, rng)
    if (mutated_good, mutated_bad) == (good, bad):
        return None
    out = {
        "sentence_good": mutated_good,
        "sentence_bad": mutated_bad,
        "rule": row_in.get("rule") or row_in.get("UID") or "mutated_blimp",
        "field": row_in.get("field", "unknown"),
        "linguistics_term": row_in.get("linguistics_term", "unknown"),
        "source": "symliquid_babylm_mutated_holdout_factory",
        "source_rule": row_in.get("rule") or row_in.get("UID"),
        "source_pair_id": row_in.get("pair_id") or row_in.get("pairID"),
        "source_path": row_in.get("source_path"),
        "source_hash": pair_key(good, bad),
        "mutation_kind": "lexical_feature_preserving",
    }
    return out


def mutate_sentence(sentence: str, rng: random.Random) -> str:
    tokens = sentence.split()
    changed = False
    out_tokens = []
    for token in tokens:
        match = re.match(r"^([^A-Za-z]*)([A-Za-z_]+)([^A-Za-z]*)$", token)
        if not match:
            out_tokens.append(token)
            continue
        left, word, right = match.groups()
        lower = word.lower()
        choices = PRESERVING_REPLACEMENTS.get(lower)
        if choices and rng.random() < 0.45:
            replacement = rng.choice(choices)
            if word[:1].isupper():
                replacement = replacement.capitalize()
            out_tokens.append(f"{left}{replacement}{right}")
            changed = True
        else:
            out_tokens.append(token)
    return " ".join(out_tokens) if changed else sentence


def build_report(
    *,
    generated: list[dict[str, Any]],
    sources: list[str],
    excludes: list[str],
    sentence_exclusion: bool,
    residual_analysis: str,
    target_rules: list[str],
    target_terms: list[str],
    out: str,
    seed: int,
) -> dict[str, Any]:
    by_rule = Counter(row.get("rule", "unknown") for row in generated)
    by_term = Counter(row.get("linguistics_term", "unknown") for row in generated)
    by_kind = Counter(row.get("mutation_kind", "unknown") for row in generated)
    rule_terms: dict[str, set[str]] = defaultdict(set)
    for row in generated:
        rule_terms[str(row.get("rule", "unknown"))].add(
            str(row.get("linguistics_term", "unknown"))
        )
    return {
        "policy": "local_only_no_external_inference",
        "methodology": "babylm_mutated_holdout_factory",
        "benchmark_family": "babylm_mutated_holdout",
        "out": out,
        "seed": seed,
        "cases": len(generated),
        "sources": sources,
        "excludes": excludes,
        "sentence_exclusion": sentence_exclusion,
        "residual_analysis": residual_analysis,
        "target_rules": target_rules,
        "target_terms": target_terms,
        "by_rule": dict(by_rule.most_common()),
        "by_linguistics_term": dict(by_term.most_common()),
        "by_mutation_kind": dict(by_kind.most_common()),
        "rule_terms": {rule: sorted(terms) for rule, terms in sorted(rule_terms.items())},
        "external_inference_calls": 0,
        "verification": {
            "jsonl_written": True,
            "minimal_pair_schema": True,
            "source_exact_pair_exclusion": True,
            "source_exact_sentence_exclusion": sentence_exclusion,
            "deterministic_seed": True,
        },
    }


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
