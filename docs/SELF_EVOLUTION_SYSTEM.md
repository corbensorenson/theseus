# Self-Evolution System

Last updated: 2026-05-15.

This document describes the new self-evolution lane for SparkStream/RMI. The goal is not to let the project mutate itself randomly. The goal is to let the local system keep improving automatically, with a sparse teacher allowed to implement small changes only when local evidence reaches a real wall.

## Principle

The system should remain as small as possible.

The intervention ladder is:

1. audit the benchmark;
2. improve data or residual-targeted synthetic bridges;
3. improve training profile or optimizer settings;
4. improve inference, tools, or routing;
5. close repeated loops into verified tools;
6. add benchmark adapters or bridge benchmarks;
7. make the smallest architecture change;
8. add parameters only when evidence says mechanism is missing.

This is encoded in `configs/self_evolution_policy.json` and `configs/architecture_search_space.json`.

## New Machinery

| Component | Purpose |
| --- | --- |
| `scripts/self_evolution_governor.py` | Reads all current evidence and decides which self-improvement lanes are ready. |
| `scripts/teacher_self_edit_runner.py` | Creates a guarded branch, calls the Codex teacher in apply mode, then runs checks. |
| `scripts/attd_analyzer.py` | Computes deterministic Assembly-Theoretic Technical Debt metrics and maintenance packets before autonomous source growth. |
| `scripts/benchmark_adapter_factory.py` | Turns governed online sources and local ROM profiles into benchmark cards and smoke plans. |
| `scripts/architecture_experiment_governor.py` | Ranks zero-param-first architecture experiments from current residuals and gates. |
| `scripts/architecture_guidance_loop.py` | Lets the sparse teacher diagnose residual clusters and emit experiment specs, without solving benchmark tasks or applying patches. |
| `scripts/cell_lifecycle.py` | Adds anti-bloat expiration, renewal, improve/split/compress, and non-destructive prune planning for arms, suckers, tools, verifiers, systems, and mastered data. |
| `scripts/grammar_suckers.py` | Refreshes deterministic rule/verifier suckers such as Python AST checks, English surface checks, and SBL routing traces. |
| `scripts/deterministic_taming_stack.py` | Summarizes verifier-layer health across Python, Rust, JS/TS, SBL, tool schemas, and memory provenance. |
| `scripts/autoresearch_gap_audit.py` | Checks the governance loop against Karpathy-style Autoresearch invariants: scoped edit surface, fixed budget, fixed metric, compact result ledger, keep/discard/crash statuses, and simplicity pressure. |
| `scripts/autoresearch_experiment_ledger.py` | Appends compact outcome rows for comparable experiments. |
| `scripts/loop_closure_harvester.py` | Finds repeated successful workflows that should become verified tools. |
| `benchmarks/cards/*.json` | Machine-readable benchmark cards for adapters, smokes, permission envelopes, and regression policy. |

The dashboard now exposes these reports through `/api/status` and shows panels for Self-Evolution, ATTD Repo Health, Architecture Experiments, Benchmark Adapter Factory, and Loop Closure.

## Cell Lifecycle / Anti-Bloat

Project Theseus now treats arms, suckers, verifier suckers, tools, systems, and
some data surfaces as lifecycle-governed cells. Each cell has usage evidence,
benchmark or readiness evidence when available, a review/expiration window, and
a decision:

```text
renew | observe | improve | split_or_compress | protect | retire
```

The current lifecycle report is `reports/cell_lifecycle.json`. It is
non-destructive by default: it can recommend improvement, split/compress, tool
creation pressure, or data archival, but it does not delete arms, tools, data,
or reports during autonomy. Any future prune action must go through a visible
plan such as `reports/cell_lifecycle_prune_plan.json` and must respect protected
cells in `configs/cell_lifecycle_policy.json`.

This is the anti-bloat counterpart to loop closure. Repeated useful workflows
can become tools; stale or weak tools get improvement pressure instead of
silently accumulating forever.

## Deterministic Rule Substrate

The grammar-sucker and deterministic-taming lanes are the rule side of the
system. They are not benchmark-answer shortcuts; they are verifiers and
structure checks around token generation:

- Python: AST parse, function shape, import guard, and sandbox-facing checks.
- Rust: cargo/rustfmt/clippy-style health when available.
- JS/TS: parser/type-checker lane as a planned sucker.
- English: surface grammar, punctuation, quote, and structure checks.
- SBL: semantic-backbone traces for routing and transfer.
- Tools/memory: schema, permission, provenance, and staleness checks.

Current machine-readable truth lives in `reports/grammar_suckers.json` and
`reports/deterministic_taming_stack.json`. These reports are allowed to reject
invalid candidates and route pressure; they must not read public benchmark
solutions or hidden public tests.

## ATTD Repo Health Gate

ATTD is the deterministic source-growth governor. It does not use a learned model or the teacher. It reads tracked code, configs, docs, benchmark cards, and dashboard files, then writes:

- `reports/attd_report.json`
- `reports/attd_maintenance_packets.json`
- `reports/attd_report.md`

The trigger state is:

| State | Meaning |
| --- | --- |
| `GREEN` | Source structure is healthy enough for normal autonomy. |
| `YELLOW` | Autonomy may continue, but maintenance packets should be worked down. |
| `RED` | Long autonomy, teacher self-edit, architecture change, and adapter-card writes are blocked except for ATTD maintenance. |

The first maintenance packet normally targets the largest or most structurally expensive module. The current baseline is intentionally calibrated so historical repository growth creates `YELLOW` pressure, while only fresh dirty residue or hard-cap violations create `RED`.

ATTD is now a first-class teacher trigger. If ATTD is `YELLOW` or `RED` and maintenance packets exist, `scripts/self_evolution_governor.py` marks the guarded teacher lane ready with `primary_reason=attd_maintenance`. In that mode the teacher is not asked to invent new architecture; it is asked to consume the highest-priority packets with the smallest verified simplification that lowers debt pressure or contains the hotspot while preserving behavior.

## Guarded Teacher Self-Edit

The teacher is allowed to write code only through this lane:

```text
local evidence
  -> architecture_experiment_governance says architecture change is justified
     or ATTD maintenance packets require repo-health repair
  -> self_evolution_governance says teacher apply is allowed
  -> clean worktree check, with optional auto-checkpoint commit
  -> create codex/self-evolution/<timestamp> branch
  -> teacher apply patch
  -> local checks
  -> branch remains for review/merge
```

The runner never uses destructive reset or checkout to hide failed patches. If checks fail, it leaves the branch and reports the exact failure.

Default checks include:

- Python compile of the autonomy/teacher/evolution scripts;
- external inference audit;
- candidate promotion gate smoke;
- launch readiness smoke.

The teacher is not supposed to train, bulk-download data, score benchmarks, use hidden external inference, or add architecture by vibes. It receives evidence files and is asked for the smallest patch that addresses the measured wall. For `attd_maintenance`, the evidence is `reports/attd_report.json` plus `reports/attd_maintenance_packets.json`.

Every guarded teacher repair appends a compact distillation trace to `reports/teacher_self_edit_traces.jsonl` and a workflow trace to `reports/workflow_routing_traces.jsonl`. Those traces record the reason, selected arms, ATTD before/after state, checks, changed files, and teacher response summary. `scripts/loop_closure_harvester.py` reads those traces so repeated successful teacher repairs can eventually become a local verified maintenance tool.

## Benchmark Adapter Factory

The adapter factory makes new sources routable:

```text
online source catalog / local ROM registry
  -> benchmark card
  -> loader/scorer smoke plan
  -> contamination policy
  -> permission envelope
  -> regression policy
```

Cards are written under `benchmarks/cards/`. They do not include ROM bytes or private asset hashes. Local ROM cards only reference a profile; runtime resolution uses ignored local registries.

## Architecture Experiment Governance

Architecture experiments are ranked under the small-model rule. Current high-priority families are:

- residual bridge data;
- loop-closure tooling;
- benchmark adapter factory;
- Rust/CUDA hot-loop ownership;
- learned sequence-state probes;
- residual adapters;
- KAN-lite readout sweeps only after cheaper steps fail.

`reports/architecture_experiment_governance.json` records whether a real architecture change is allowed and what the next experiment should be.

## Autoresearch-Style Outcome Ledger

Karpathy's `autoresearch` loop is intentionally blunt: run a fixed-budget experiment, record commit, metric, memory, status, and description, then keep or discard based on evidence. SparkStream keeps richer RMI ledgers, but it now also keeps that compact view:

- `configs/autoresearch_loop_policy.json`
- `reports/autoresearch_gap_audit.json`
- `reports/autoresearch_experiment_ledger.jsonl`
- `reports/autoresearch_experiment_ledger_summary.json`

The audit does not replace the RMI gates. It checks that our autonomous experiments stay comparable: fixed profile budgets, explicit edit boundaries, baseline row present, keep/discard/crash statuses defined, ATTD simplicity pressure active, and resource cost recorded.

## Loop Closure

`scripts/loop_closure_harvester.py` scans workflow traces and autonomy ledgers for repeated success. Candidate tools are not automatically trusted; each receives parameters, preconditions, verification plan, runtime tier, risk tier, and retirement criteria.

This is how the system stops re-improvising the same workflows forever.

## Current Expected Behavior

During an autonomy cycle, SparkStream now refreshes:

- ATTD repo-health governance;
- benchmark adapter cards;
- architecture experiment queue;
- loop-closure candidates;
- self-evolution governance;
- guarded teacher self-edit decision.

If the worktree is dirty, `scripts/teacher_self_edit_runner.py` now follows
`guarded_self_edit.auto_commit_dirty_worktree`: it stages the current worktree,
commits a checkpoint with `auto_commit_message`, then re-checks cleanliness
before creating the self-evolution branch. If that automatic checkpoint fails,
teacher apply remains blocked so human/Codex changes are not silently mixed with
autonomous edits. This is a safety boundary, not a capability boundary.

Before teacher prompts are written, the runner refreshes
`reports/personality_context_last.json` through
`scripts/personality_context_builder.py` and includes it with drift/belief
governance reports as evidence. Teacher patches should preserve truth before
compliance, agency before convenience, least sufficient power, and belief-update
governance.

## Source Of Truth

Machine-readable truth:

- `reports/self_evolution_governance.json`
- `reports/attd_report.json`
- `reports/attd_maintenance_packets.json`
- `reports/architecture_experiment_governance.json`
- `reports/autoresearch_gap_audit.json`
- `reports/autoresearch_experiment_ledger.jsonl`
- `reports/autoresearch_experiment_ledger_summary.json`
- `reports/benchmark_adapter_factory.json`
- `reports/loop_closure_harvester.json`
- `reports/teacher_self_edit_last.json`
- `reports/teacher_self_edit_traces.jsonl`

Human-readable companion reports:

- `reports/self_evolution_governance.md`
- `reports/attd_report.md`
- `reports/architecture_experiment_governance.md`
- `reports/benchmark_adapter_factory.md`
- `reports/loop_closure_harvester.md`
