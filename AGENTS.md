# Project Theseus Operating Charter

North star: build a private, locally trained model Corben uses daily, with
zero external inference at serving time and a teacher bill that trends to zero.

## Hard Rules

1. External inference is permitted only as a governed teacher during training.
   Externally generated tokens are never served to a user.
2. Public benchmarks are calibration only. Never train on public benchmark
   prompts, tests, hidden tests, solutions, traces, or answer templates.
3. Public benchmarks stay calibration-only, but fresh frozen measurement
   surfaces are authorized through the governed run registry by default. Do
   not rerun an exact consumed surface or bypass contamination checks.
4. Teacher-generated training rows are allowed only through the governed
   teacher-distillation gate. They must be retained, provenance-tagged,
   license-checked, verifier-accepted, leakage-audited, and never routed to
   runtime serving.
   Static third-party corpora are data sources rather than live teacher calls:
   openly licensed model-derived rows may be admitted through the corpus gate
   when provenance, quality tier, synthetic share, deduplication, contamination,
   retention, and permitted use are explicit. This does not authorize that
   provider for interactive teacher or runtime inference.
5. Do not add arbitrary remote execution, public gateway operation, or
   unbounded self-update behavior.
6. High-quality static open-source/model-derived training data is eligible
   regardless of provider when its license permits training and provenance,
   quality, deduplication, contamination, synthetic-share, and retention checks
   pass. Static corpus admission does not grant live-teacher authority.
7. The current natural-language scope is English only. The programming-language
   scope is Python, JavaScript/TypeScript, HTML/CSS, and Rust. Non-English natural
   language is excluded or quarantined for this seed.
8. Live teacher data is targeted residual pressure, not the bootstrap corpus.
   Enforce the configured accepted-row and optimizer-sampling caps and drive both
   toward zero; bulk teacher generation is forbidden.
9. Live governed teachers must be OpenAI models accessed through ChatGPT,
   Codex, or an explicitly approved OpenAI API path. Do not invoke Anthropic or
   Claude through a paid account, API, CLI, desktop app, or project automation.
   Already-published static corpora may contain Anthropic/Claude-derived rows;
   they are eligible only as third-party data under Rules 4, 6, 7, and 8, with
   no provider credentials, subscription usage, or live generation. Provider
   provenance must remain explicit and does not waive license, quality,
   deduplication, contamination, synthetic-share, retention, or verifier gates.

## Anti-Cheating Guardrail

All capability claims require blind information-flow evidence. Generation and
ranking paths may see only the natural-language prompt, the callable signature,
and explicitly allowed runtime context. They may not see answer-identifying
metadata, including `category`, `solution`, `solution_expr`, `solution_body`,
`tests`, `hidden_tests`, `expected`, `answer`, `source_task_id`, benchmark card
labels, answer-family labels, or decoder fields derived from the hidden target
such as `return_shape`, `type_family`, or `required_constructs`.

Candidate integrity must be recomputed by an independent audit. Do not trust
candidate-emitted flags for family, fallback/template status, learned status,
promotion eligibility, public-data use, or contamination status.

Hand-authored action catalogs, deterministic renderers, fixed templates,
fallbacks, and tool calls are useful baselines/tools, but they are never learned
generation and cannot support learned-generation promotion claims.

Any report violating this rule must be marked invalid. Do not repair the
violation with wording, a new label, a nearby green report, or a broader claim.

## Current Priorities

1. Neural seed track: build independently trained English, Python, JS/TS,
   HTML/CSS, and Rust arms behind the governed Octopus/MoECOT route contract.
   Keep the mixed dense transformer as a matched falsification control. Compare
   arm, route, composition, and answer behavior separately before claiming the
   modular substrate mattered.
2. Teacher distillation track: make
   `teacher_share_of_accepted_training_rows` a durable ledger metric and drive
   that share down as verified self-generated data takes over.
3. Dogfood track: make one daily-use lane genuinely useful to Corben and log
   whether it was used, missed, or ignored.
4. Repo health track: consolidate scripts/docs, retire superseded material, and
   keep a plain-English glossary for project terms.

## Breadth Freeze

Do not create new lanes, dashboards, product surfaces, mobile/spatial features,
benchmark families, or doc families unless they directly serve a priority
above. Maintenance and regression fixes are allowed.

Do not generate more private ecology, shadow, or residual-frontier suites just
to produce another private 1.0. Current governance already reports
`no_private_frontier_action_remaining`; new training pressure must come from
real traces, governed teacher distillation, or the matched neural seed
experiment.

## Discovery Track

The SymLiquid/CGS/VSA/liquid-substrate bet is protected. The survival lane may
use known-good controls, but it may not absorb the discovery lane. Experiments
must report matched compute, matched data, verifier results, cost, and residuals
so a win or loss is attributable.

## Operating Posture

Every change should trace to one current priority. Prefer deleting,
consolidating, and making evidence easier to audit over adding another report.
A report is evidence, not progress. Progress means the student improves, the
harness becomes more honest, or Corben uses the system more.

When blocked, write the wall plainly and stop. Do not manufacture a nearby
green artifact to avoid the blocker.
