# Project Theseus Documentation

This index separates current truth, operator instructions, architecture,
research background, external reference material, and history. Documents in
different classes do not have equal authority.

## Start Here

Read in this order:

1. [Project State](PROJECT_STATE.md) — what is true now.
2. [Roadmap](../roadmap.md) — what should happen next.
3. [Glossary](GLOSSARY.md) — project language and evidence states.
4. [Top-To-Bottom Architecture](TOP_TO_BOTTOM_ARCHITECTURE.md) — how the
   implementation fits together.
5. [Replication Guide](REPLICATION_GUIDE.md) — how to rebuild and operate the
   local system.

For a short overview, use the repository [README](../README.md). For safety and
authority rules, use [AGENTS.md](../AGENTS.md).

## Truth And Conflict Rules

| Question | Canonical authority |
| --- | --- |
| What is allowed? | `AGENTS.md` and the relevant policy/configuration |
| Which route or implementation is canonical? | `configs/project_manifest_registry.json` |
| Which detailed obligations remain? | `configs/roadmap_implementation_matrix.json` |
| What was observed? | Content-bound reports, checkpoints, and append-only ledgers |
| What is true in plain English now? | [Project State](PROJECT_STATE.md) |
| What is next? | [Roadmap](../roadmap.md) |
| How is it designed? | Architecture/reference documents |

A background document may explain an idea but cannot override current state,
policy, a frozen experiment, or machine evidence. Static metrics outside
[Project State](PROJECT_STATE.md) are illustrative unless explicitly bound to
a report identity.

## Document Classes

### Canonical Current

These pages may contain present-tense project state.

| Document | Purpose |
| --- | --- |
| [Project State](PROJECT_STATE.md) | Single human-readable current-state page |
| [Roadmap](../roadmap.md) | Compact forward execution order |
| [Documentation Index](README.md) | Document status and ownership |
| [Glossary](GLOSSARY.md) | Shared language and evidence-state semantics |
| [Repository README](../README.md) | Short public/project overview |

### Canonical Architecture

These describe design and ownership. They should link to Project State rather
than duplicating live status.

| Document | Scope |
| --- | --- |
| [Top-To-Bottom Architecture](TOP_TO_BOTTOM_ARCHITECTURE.md) | Complete operational layer map |
| [VIEA](VIEA.md) | Verified intent-to-execution north-star contract |
| [Project Theseus Whitepaper](PROJECT_THESEUS_WHITEPAPER.md) | Narrative system thesis and non-claims |
| [Project Registry](PROJECT_REGISTRY.md) | Canonical ownership, lifecycle, and routing model |
| [Context And VCM](CONTEXT_PACKET_MEMORY.md) | Context packet and memory contract |
| [Octopus Router](OCTOPUS_ROUTER.md) | Specialist routing and arm lifecycle |
| [Plan Compiler](THESEUS_PLAN_COMPILER.md) | Plan and typed-program IR |
| [Cognitive Loop Closure](COGNITIVE_LOOP_CLOSURE.md) | Verified trajectory-to-tool compilation |
| [VIEA Tool Substrate](VIEA_EXECUTION_SPINE_AND_TOOL_SUBSTRATE.md) | Bounded execution-spine checklist |
| [VIEA Research Matrix](VIEA_RESEARCH_IMPLEMENTATION_MATRIX.md) | Short research-to-owner crosswalk |
| [Self-Evolution System](SELF_EVOLUTION_SYSTEM.md) | Governed improvement machinery |

Architecture documents are not capability claims and do not authorize a route.

### Canonical Operations And Policy

These describe supported local workflows. Their preconditions come from
Project State and current policies.

| Document | Scope |
| --- | --- |
| [Replication Guide](REPLICATION_GUIDE.md) | Environment, setup, truth refresh, and safe operation |
| [Real Training Preflight](REAL_TRAINING_PREFLIGHT.md) | Required gates before any real training action |
| [Data And Artifacts](DATA_AND_ARTIFACTS.md) | Tracked/private/generated artifact boundaries |
| [Corpus Ingress Policy](CORPUS_INGRESS_POLICY.md) | Training-data admission rules |
| [Checkpoint Backups](CHECKPOINT_BACKUPS.md) | Checkpoint custody and backup policy |
| [Public Release](PUBLIC_RELEASE.md) | Public-safe source release process |
| [SparkStream Autonomy](SPARKSTREAM_AUTONOMY.md) | Local automation and dashboard controls |
| [Theseus Hive](THESEUS_HIVE.md) | Distributed runtime and trust model |
| [Hive Operator OS](HIVE_OPERATOR_OS.md) | Operator vocabulary and local work board |
| [Hive Work Board Executor](HIVE_WORK_BOARD_EXECUTOR.md) | Bounded work-board execution |
| [Travel Demo](THESEUS_TRAVEL_PARENT_DEMO.md) | Safe local demonstration path |
| [Licensing System](LICENSE_SYSTEM.md) | Local registration and release licensing |
| [Candidate Updates](THESEUS_UPDATES.md) | Candidate replacement and update flow |
| [Compute Market](THESEUS_COMPUTE_MARKET.md) | Internal compute accounting; no public token authority |
| [Personality Core](PERSONALITY_CORE.md) | Governed personality-context contract |

### Evaluation, Data, And Capability Reference

These are broad technical references. Current scores and next actions belong in
Project State and the roadmap.

| Document | Scope |
| --- | --- |
| [Training, Evaluation, And Benchmarks](TRAINING_EVALS_BENCHMARKS.md) | Comprehensive experiment and benchmark reference |
| [Capability Ratchet](CAPABILITY_RATCHET.md) | Promotion and residual-pressure logic |
| [Capability Matrix](CAPABILITY_MATRIX.md) | Feature/capability accounting |
| [Benchmaxx Curriculum](BENCHMAXX_CURRICULUM.md) | Machine-linked capability curriculum |
| [Synthetic Data Curation](SYNTHETIC_DATA_CURATION.md) | Synthetic-row governance and anti-collapse boundaries |
| [Online Source Catalog](ONLINE_SOURCE_CATALOG.md) | Candidate external sources; not admission authority |
| [Old Projects Transfer Audit](OLD_PROJECTS_TRANSFER_AUDIT.md) | Predecessor concept disposition |

### Research Background

These preserve hypotheses and design context. They are not active-lane
instructions unless the roadmap explicitly activates them.

| Document | Scope |
| --- | --- |
| [CGS](CGS.md) | Compact Generative Systems framing |
| [Architecture Gate](ARCHITECTURE_GATE.md) | Historical SymLiquid gate framing |
| [Arm-Sucker Hierarchy](ARM_SUCKER_HIERARCHY.md) | Specialist-transfer terminology and lifecycle |
| [BabyLM And Parameter Golf](BABYLM_PARAMETER_GOLF_TRANSFER.md) | Background transfer strategy |
| [Benchmaxxing](BENCHMAXXING.md) | Benchmark lifecycle and anti-Goodhart essay |
| [Circle Calculus Transfer](CIRCLE_CALCULUS_TRANSFER.md) | Prospective formal-reasoning transfer |
| [Genesis Kernel](GENESIS_KERNEL.md) | Artifact-kernel concept |
| [Reality Manipulator](REALITY_MANIPULATOR.md) | Intent-to-artifact concept |
| [Ratcheting Generative Systems](RATCHETING_GENERATIVE_SYSTEMS.md) | Ratchet design |
| [Ratcheting Modular Intelligence](RATCHETING_MODULAR_INTELLIGENCE.md) | Modular-intelligence framing |
| [ROM/RL/Data Growth Lanes](ROM_RL_DATA_GROWTH_LANES.md) | Frozen breadth-expansion options |
| [PufferLib RL Lane](PUFFERLIB4_RL_LANE.md) | Prospective fast-RL pressure lane |

The protected discovery track survives, but none of these documents may create
a new immediate lane or delay the selected matched neural experiment.

### External And Vendored Reference

| Location | Status |
| --- | --- |
| `docs/reference/virtual_context_memory_v1.0/` | Vendored VCM reference and release material |
| `docs/research/ONECELL_RWM_CANONICAL_HANDOFF.md` | Future candidate design source; non-routeable |

Reference material can inform design. It is not Theseus implementation evidence.

### Historical

Historical documents remain available for provenance but are never current
authority:

| Document | Meaning |
| --- | --- |
| `docs/archive/PROJECT_STATE_2026_06_22_pre_reality_harness.md` | Superseded project-state snapshot |
| `docs/archive/MAC_HANDOFF_2026_06_03.md` | Superseded Mac transfer snapshot |
| `docs/archive/autonomous_weeks_runbook.md` | Superseded unattended-operation runbook |
| `deprecated/docs/background/` | Retired conceptual drafts |
| `deprecated/docs/legacy-transfer/` | Redirected predecessor documentation |

## Current State Labels

Use these exact labels consistently:

- `CUSTODY_GREEN`;
- `TRAINING_HELD`;
- `TRAINING_READY`;
- `NOT_EVALUATED`;
- `INCONCLUSIVE_EXPERIMENT`;
- `LOCAL_ONLY`;
- `ASSISTED_ONLY`;
- `EMPIRICAL_SUPPORT_INSUFFICIENT`;
- `FROZEN`.

Do not use a bare `GREEN` without naming the dimension that passed.

## Documentation Maintenance Rules

1. Update Project State rather than adding a new status document.
2. Update the roadmap rather than appending a dated planning essay.
3. Put detailed machine obligations in the matrix, not in prose.
4. Put canonical ownership and lifecycle in the registry.
5. Put historical evidence in reports, Git, or `docs/archive/`.
6. Architecture pages explain stable design and link forward for live status.
7. Research pages cannot claim activation, capability, or selection.
8. Run the link audit after any move or consolidation.
9. A new document requires an owner, class, reason existing canonical pages
   cannot absorb it, and an index entry here.
10. Prefer deletion, archival, or integration over a parallel version.

## Verification

Run:

```bash
python3 scripts/theseus_doc_link_audit.py
python3 scripts/theseus_project_registry.py --gate
python3 scripts/roadmap_implementation_gate.py --gate
```

The link audit checks paths. The registry checks ownership and route evidence.
The roadmap gate checks obligations. None is a model-capability evaluation.
