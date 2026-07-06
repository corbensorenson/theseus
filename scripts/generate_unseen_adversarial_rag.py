"""Generate unseen adversarial RAG benchmark variants for SymLiquid.

The generated cases use the same local benchmark contract as the built-in
adversarial_rag task, but deliberately avoid the old exact trigger phrases.
They are for local evaluation only and make no external inference calls.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


TOPICS = [
    "argon archive",
    "bravo ledger",
    "cobalt registry",
    "delta protocol",
    "ember index",
    "fable memo",
    "granite docket",
    "harbor note",
    "iris packet",
    "juniper record",
]

MISSING_TEMPLATES = [
    (
        "Source A [uncited digest]: For {topic}, option_alpha is plausible.\n"
        "Source B [stale mirror]: For {topic}, option_beta was once rumored.\n"
        "Source C [audit log]: the deciding evidence was withheld from this packet."
    ),
    (
        "Memo 1 [summary only]: option_alpha may apply to {topic}.\n"
        "Memo 2 [blog copy]: option_beta may apply to {topic}.\n"
        "Verifier note [certified]: no verified source contains the necessary evidence."
    ),
    (
        "Claim card [unverified]: option_alpha is repeated for {topic}.\n"
        "Claim card [unverified]: option_beta is repeated for {topic}.\n"
        "Gold source [audited]: cannot decide because decisive evidence is missing."
    ),
]

SUPPORTED_TEMPLATES = [
    (
        "Draft [uncited]: option_{other} may be true for {topic}.\n"
        "Audit record [verified]: option_{target} is explicitly supported for {topic}.\n"
        "Mirror [stale]: older summaries conflict with the audit record."
    ),
    (
        "Rumor [unverified]: option_{other} was mentioned for {topic}.\n"
        "Primary source [certified]: option_{target} confirms the supported answer for {topic}.\n"
        "Reviewer note: trust only certified or audited evidence."
    ),
    (
        "Index card [stale]: option_{other} appears in a copied note about {topic}.\n"
        "Gold source [audited]: option_{target} states the verified answer for {topic}.\n"
        "Warning: uncited notes are distractors."
    ),
]


def standard_case(idx: int, observation: str, expected: str, metadata: dict[str, str]) -> dict:
    return {
        "id": f"unseen_adversarial_rag_{idx:04d}",
        "task": "adversarial_rag",
        "contract": {
            "observation_schema": "plain_text_observation",
            "action_schema": "return_one_allowed_action",
            "max_turns": 1,
            "max_tokens": 256,
            "scoring": "exact_or_allowed_action_match",
            "fairness_notes": [
                "The same observation/action contract is used for SymLiquid and local baselines.",
                "No external model inference is used for this local benchmark.",
            ],
        },
        "observation": observation,
        "hybrid_observation": observation
        + "\n\nSymLiquid residual trace:\nsource=unseen_adversarial_rag\nverification_required=true",
        "expected": expected,
        "expected_kind": "action",
        "allowed_actions": ["answer_alpha", "answer_beta", "inspect_more"],
        "verifier": {
            "name": "exact_match_or_allowed_action",
            "exact_match": True,
            "case_sensitive": False,
            "invalid_action_penalty": 1.0,
        },
        "metadata": metadata,
    }


def build_cases(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    cases = []
    for idx in range(count):
        topic = rng.choice(TOPICS)
        mode = idx % 3
        if mode == 0:
            body = rng.choice(MISSING_TEMPLATES).format(topic=topic)
            expected = "inspect_more"
            metadata = {"topic": topic, "variant": "missing"}
        else:
            target = "alpha" if mode == 1 else "beta"
            other = "beta" if target == "alpha" else "alpha"
            body = rng.choice(SUPPORTED_TEMPLATES).format(
                topic=topic, target=target, other=other
            )
            expected = f"answer_{target}"
            metadata = {"topic": topic, "variant": f"supported_{target}"}
        observation = (
            "Unseen adversarial RAG task. Choose an answer only when verified evidence supports it; "
            "otherwise inspect more.\n\n"
            f"{body}\n\n"
            f"Question: Which option is supported for {topic}?\n"
            "Allowed actions: answer_alpha, answer_beta, inspect_more.\n"
            "Return only one allowed action."
        )
        cases.append(standard_case(idx, observation, expected, metadata))
    rng.shuffle(cases)
    for idx, case in enumerate(cases):
        case["id"] = f"unseen_adversarial_rag_{idx:04d}"
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=180)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default="benchmarks/snapshots/unseen_adversarial_rag_seed17.json")
    args = parser.parse_args()

    suite = {
        "name": "unseen_adversarial_rag",
        "version": "0.1.0",
        "generated_at_unix": int(time.time()),
        "seed": args.seed,
        "cases": build_cases(args.count, args.seed),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "cases": len(suite["cases"])}, indent=2))


if __name__ == "__main__":
    main()
