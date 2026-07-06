# Theseus Personality Core

The personality core is a governed local memory layer built from
`personality-documents/`. It is meant to capture the user's best durable
patterns without reducing the system to one large prompt.

Current state, 2026-05-15:

```text
runtime audit: GREEN
documents used: 5
activation-eligible snippets: 905
retrieval cards: 96
selected runtime cards: 8
drift average score: 1.0
multi-turn conversation benchmark: GREEN
quarantined belief updates: 0
external inference calls: 0
training use: manifest-only until explicitly approved
```

The personality core is now a live runtime substrate. It is not merely a
generated report.

## Research Basis

The implementation follows a layered approach:

- Memory first: Generative Agents showed that believable long-horizon behavior
  depends on a memory stream, reflection, and retrieval instead of stuffing all
  experience into a prompt.
- Trainable persona later: Character-LLM frames role/person simulation as
  profile plus experience reconstruction plus training, which maps here to a
  future manifest and adapter lane.
- Efficient adapters when ready: LoRA-style fine-tuning can add personal style
  or preference behavior without full model retraining, but only after Theseus
  has a selected transformer training surface and eval gates.
- Steering as an advanced lane: persona-vector work suggests activation-space
  trait monitoring and steering, but Theseus should only use this when the
  runtime exposes calibrated model internals.
- Personalized alignment needs guardrails: recent surveys emphasize preference
  memory, personalized generation, feedback, and ethical boundaries together.

Primary references:

- https://arxiv.org/abs/2304.03442
- https://arxiv.org/abs/2310.10158
- https://arxiv.org/abs/2106.09685
- https://arxiv.org/abs/2507.21509
- https://arxiv.org/abs/2503.17003
- https://arxiv.org/abs/2309.11696

## Local Artifacts

- `configs/personality_core_policy.json`: source, privacy, best-self, reality
  contract, anti-drift, and adaptation-lane policy.
- `scripts/personality_core.py`: distills local documents into reports.
- `scripts/personality_context_builder.py`: the blessed runtime loader for
  compact core, relevant retrieval cards, best-self contract, reality contract,
  and anti-drift rules.
- `scripts/personality_runtime_audit.py`: integration proof that the core is
  distilled, selected by the runtime context builder, consumed by checkpoint
  chat, checked by drift evals, checked by the multi-turn conversation
  benchmark, guarded by belief governance, and enforced by launch readiness.
- `scripts/personality_drift_eval.py`: runs launch-facing drift prompts through
  checkpoint chat and scores whether answers remain faithful to the core.
- `scripts/multi_turn_conversation_benchmark.py`: runs local multi-line,
  multi-turn conversation cases through `checkpoint_chat.py` and verifies
  session continuity, correction handling, active constraint carry, and
  personality-context attachment on every turn.
- `scripts/belief_update_governor.py`: records observation-to-belief updates
  with confidence, inherited-core conflicts, and accept/review/quarantine
  decisions.
- `reports/personality_core.json`: machine-readable profile, value scores,
  voice profile, retrieval cards, and adaptation plan.
- `reports/personality_context_last.json`: selected runtime personality context
  for the latest prompt/task surface.
- `reports/personality_runtime_audit.json`: green/yellow/red proof that the
  personality core is wired into live runtime surfaces.
- `reports/personality_drift_eval.json`: drift-eval scorecard.
- `reports/multi_turn_conversation_benchmark.json`: local conversation runtime
  scorecard. Its `summary.accuracy` is consumed by Benchmaxxing as
  `conversation_multiturn`.
- `reports/checkpoint_chat_sessions/*.jsonl`: bounded session histories used by
  checkpoint chat and hive operator chat.
- `reports/belief_update_governance.json` and
  `reports/belief_update_ledger.jsonl`: durable belief update governance.
- `reports/personality_core.md`: compact human-readable summary.
- `reports/personality_core_training_manifest.jsonl`: local future-training
  rows. These are not automatically consumed.

`reports/` is ignored, so the distilled runtime memory stays local unless the
user deliberately exports it.

## Default Privacy Posture

The policy includes markdown/text documents and selected public-writing files
from Twitter/X archives: `tweets.js`, `note-tweet.js`, and `article.js`.

It excludes direct messages, ad impressions, account metadata, IP/device files,
likes, follows, Grok chats, and archive assets by default. Secret-like text,
emails, IP addresses, and sensitive tactical fragments are blocked or suppressed
from activation cards.

## Runtime Use

Checkpoint chat, Hive `checkpoint_chat`, the OpenAI-compatible shim,
autonomous goal routing, teacher self-edit packets, context packet memory,
watchdog, dashboard status, autonomy cycles, and launch readiness all route
through or verify `scripts/personality_context_builder.py`. That keeps the
worldview path centralized instead of letting each surface read raw personality
reports differently.

Checkpoint chat is now session-aware. The hive operator can pass
`session_id`; `checkpoint_chat.py` records bounded JSONL turns under
`reports/checkpoint_chat_sessions/` and renders recent history into the next
turn. This is still a grounded report/chat shim, not the final learned
conversation model, but it gives Theseus a real continuity contract and a
benchmark target.

Refresh and prove the runtime path with:

```powershell
python scripts\personality_runtime_audit.py --refresh --out reports\personality_runtime_audit.json
python scripts\multi_turn_conversation_benchmark.py --out reports\multi_turn_conversation_benchmark.json --markdown-out reports\multi_turn_conversation_benchmark.md
python scripts\autonomy_launch_readiness.py --profile inner_loop --out reports\autonomy_launch_readiness.json
```

The current launch-readiness gate blocks long autonomy if the personality
context or drift eval is missing. Checkpoint chat attaches a compact
`personality_context` block to every response, not only responses about the
personality core.

The core should be treated as a corrigible map of the user's best patterns, not
as permission to reproduce every historical impulse in the source corpus.

## Reality Contract

The runtime personality context separates five layers:

- Style and voice: how Theseus should sound and collaborate.
- Values: what it should protect and optimize for.
- Epistemology: how it should orient to truth, uncertainty, and correction.
- Metaphysics and speculation: worldview language that can inspire hypotheses
  but cannot override evidence, consent, or safety.
- Hard safety invariants: rules that must not drift silently.

Observation-driven growth flows through `scripts/belief_update_governor.py`.
Each candidate update records the observation, inferred belief, confidence,
conflict with inherited core, and decision: accepted, needs review, or
quarantined.
