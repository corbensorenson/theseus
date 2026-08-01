# Project Theseus Operating Charter

North star: use Theseus to rigorously test and earn evidence for the largest,
highest-leverage ideas in *The ASI Stack*. The proving system must be private
and locally served, use adequate matched experiments, and drive external
teacher dependence toward zero. Autonomous machine-verifiable work is an
experimental substrate, not the primary product goal.

## Hard Rules

1. External inference is permitted only as a governed teacher during training
   or as a prospectively sealed OpenAI measurement-only reference control on
   public, license-compatible experimental tasks. Reference-control outputs
   are never served to a user, admitted as training rows, used to select or
   tune tasks, mixed into local-model denominators, or granted source effects.
   They must use the same candidate-visible task packet and independent hidden
   evaluator as the corresponding local arms, with provider/model/effort,
   wrapper, budgets, cost, and information-flow differences reported.
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
   Measurement-only reference controls inherit the same OpenAI-only provider
   restriction and may run only through ChatGPT, Codex, or an explicitly
   approved OpenAI API path.

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

## Negative-Evidence Guardrail

A negative result is scoped to the exact implementation, data, scale, objective,
optimizer, budget, evaluator, and operating regime that produced it. A mechanics
canary, toy proxy, hand-authored corpus, rule-based stand-in, linear probe, missing
training stage, underpowered run, or evaluator with known construct-validity gaps
cannot falsify the full mechanism it approximates.

Before a result may retire or broadly falsify an architecture, an independent
adequacy audit must establish all of the following:

1. The implementation faithfully exercises every causal mechanism named in the
   claim and is not a deliberately simplified proxy.
2. The candidate passes learnability, gradient-flow, overfit, checkpoint/reload,
   intervention, and ablation sanity checks appropriate to the mechanism.
3. Baselines are strong and receive matched raw data, total training compute,
   tuning opportunity, inference budget, verifier budget, and total-system cost.
4. The evaluation has adequate task diversity, multiple seeds, uncertainty or
   confidence intervals, weak-tail reporting, and power to detect the predeclared
   useful effect.
5. Heldouts, evaluators, and integrity checks are source-disjoint and independent
   of the implementation and training-data producer.

If any condition is missing, record `INCONCLUSIVE_IMPLEMENTATION` or
`INCONCLUSIVE_EXPERIMENT`, preserve the evidence, and repair the owner. Campaign
exclusion for cost or sequencing is an engineering disposition, not scientific
falsification. Never translate "this proxy failed" into "this idea failed."

## Current Priorities

1. ASI Stack proof track: select the highest-leverage book mechanisms, bind
   each to an exact falsifiable claim, exercise the real causal implementation,
   compare it with strong matched controls, and advance support only through
   source-disjoint independent evidence.
2. Autonomous work track: maintain one bounded local lane that supplies
   machine-verifiable tasks from governed, license-compatible online sources.
   This lane exists to make subsystem experiments valid; task selection,
   labeling, acceptance, and scheduling may not depend on Corben.
3. Neural seed track: build independently trained English, Python, JS/TS,
   HTML/CSS, and Rust arms behind the governed Octopus/MoECOT route contract.
   Keep the mixed dense transformer as a matched falsification control. Compare
   arm, route, composition, and answer behavior separately before claiming the
   modular substrate mattered.
4. Teacher and repo health track: make
   `teacher_share_of_accepted_training_rows` a durable ledger metric and drive
   that share down as verified self-generated data takes over; consolidate
   scripts/docs, retire superseded material, and keep a plain-English glossary.

## Breadth Freeze

Do not create new lanes, dashboards, product surfaces, mobile/spatial features,
benchmark families, or doc families unless they directly serve a priority
above. Maintenance and regression fixes are allowed.

Do not generate more private ecology, shadow, or residual-frontier suites just
to produce another private 1.0. Current governance already reports
`no_private_frontier_action_remaining`; new training pressure must come from
governed autonomous traces, governed teacher distillation, or the matched
neural seed experiment.

## Discovery Track

The SymLiquid/CGS/VSA/liquid-substrate bet is protected. The survival lane may
use known-good controls, but it may not absorb the discovery lane. Experiments
must report matched compute, matched data, verifier results, cost, and residuals
so a win or loss is attributable.

## Operating Posture

Every change should trace to one current priority. Prefer deleting,
consolidating, and making evidence easier to audit over adding another report.
A report is evidence, not progress. Progress means a major ASI Stack claim gets
stronger causal evidence, an inadequate approximation is repaired, the student
improves under a fair matched comparison, or the harness becomes more honest.

Forward execution must not wait for Corben to provide tasks, labels,
acceptance decisions, routine approvals, or timing choices. Encode bounded
authority, resource ceilings, stop conditions, and promotion rules in
machine-readable policy. If an action is outside that policy—such as external
publication, destructive evidence deletion, or unbudgeted spending—fail closed
and record the wall instead of turning ordinary research execution into a user
gate.

When blocked, write the wall plainly and stop. Do not manufacture a nearby
green artifact to avoid the blocker.
