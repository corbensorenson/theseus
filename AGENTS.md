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
5. Do not add arbitrary remote execution, public gateway operation, or
   unbounded self-update behavior.

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

1. Neural seed track: build a small student proposer behind the existing
   verifier/fanout/STS harness. Compare the SymLiquid substrate against a
   parameter- and compute-matched control arm before claiming the substrate
   mattered.
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
