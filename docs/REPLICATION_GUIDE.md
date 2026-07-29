# Project Theseus Replication Guide

Last consolidated: 2026-07-29.

This guide rebuilds the public-safe source and local control plane. Private
corpora, checkpoints, runtime ledgers, user traces, and decisive evidence are
not reconstructed from Git. Current state is in
[Project State](PROJECT_STATE.md); terminology is in
[Glossary](GLOSSARY.md).

## 1. What You Are Rebuilding

Project Theseus has five practical planes:

| Plane | Purpose | Canonical owners |
| --- | --- | --- |
| Neural experiment | MoECOT candidate, matched dense controls, MLX training, exact custody | `configs/moecot_language_arm_training.json`, `scripts/neural_seed_training_campaign.py` |
| Intent and runtime | VIEA, SCF, Octopus, local assistant, deterministic tools | `scripts/theseus_assistant_runtime.py`, `scripts/theseus_cli.py` |
| Memory and evidence | VCM, artifact graph, candidate integrity, consumption ledgers | `scripts/virtual_context_memory.py`, `scripts/blind_runtime_guard.py` |
| Governance | Data/teacher policy, authority, security, registry, roadmap gates | `AGENTS.md`, `configs/project_manifest_registry.json`, `configs/roadmap_implementation_matrix.json` |
| Discovery | SymLiquid/CGS/VSA/liquid substrate and bounded comparators | `crates/symliquid-*` |

SparkStream and Hive are operational shells around these planes. They do not
own model capability or scientific truth.

## 2. Supported Environments

The repository contains:

- Apple Silicon MLX/Metal training and inference;
- cross-platform Python control-plane code;
- Rust CPU reference crates;
- optional CUDA crates and Windows/Hive packaging;
- Linux public-safe CI;
- guarded Mac MLX qualification.

Not every host supports every lane. A replicated source checkout can validate
governance, deterministic tests, Rust logic, and the public capsule without
private model/data artifacts.

Recommended tools:

- Git;
- Python 3.12;
- Rust stable with rustfmt and clippy;
- Apple Silicon plus the pinned local MLX environment for the active campaign;
- optional CUDA host for CUDA-specific qualification.

Do not bulk-copy private corpora, user traces, credentials, or checkpoints into
the tracked tree.

## 3. Clone And Inspect

```bash
git clone https://github.com/corbensorenson/theseus.git
cd theseus
git status --short
```

Read:

```text
AGENTS.md
docs/PROJECT_STATE.md
roadmap.md
docs/GLOSSARY.md
docs/TOP_TO_BOTTOM_ARCHITECTURE.md
```

The public repository may lag the private/local operational state. Never infer
checkpoint availability or training authority from source alone.

## 4. Python And Rust Checks

Install the public-safe test tools:

```bash
python3 -m pip install pytest==8.3.5 ruff==0.15.0
```

Run the scoped public-safe Python slice defined in
`.github/workflows/ci.yml`. For local broad checks:

```bash
python3 -m pytest -q
```

Some tests are hardware-, private-artifact-, or environment-guarded. A skip is
not a pass; read its reason.

Run Rust checks:

```bash
cargo fmt --all -- --check
cargo check --workspace --locked --exclude symliquid-metal --exclude symliquid-cli
cargo clippy --workspace --all-targets --locked \
  --exclude symliquid-metal --exclude symliquid-cli -- -D warnings
cargo test --workspace --locked \
  --exclude symliquid-metal --exclude symliquid-cli
```

Mac- and CUDA-specific crates require their qualified hosts.

## 5. Public Reproducibility Capsule

The tiny capsule verifies evidence protocol rather than capability:

```bash
python3 scripts/public_reproducibility_capsule.py \
  --out-dir /tmp/theseus-public-repro \
  --gate
```

It deterministically trains a tiny authored fixture, checkpoints model,
optimizer, and cursor state, resumes exactly, emits a candidate packet, and
verifies artifact digests. It contains no private row, public benchmark,
teacher call, user trace, or production performance claim.

## 6. Refresh The Control-Plane Truth

Run:

```bash
python3 scripts/repository_license_gate.py --gate
python3 scripts/theseus_doc_link_audit.py
python3 scripts/theseus_project_registry.py --gate
python3 scripts/roadmap_implementation_gate.py --gate
```

Interpretation:

- registry GREEN means canonical routing evidence is internally consistent;
- roadmap YELLOW may represent intentionally partial or frozen phases;
- neither state is learned capability;
- `--require-pre-training-ready` is a separate stricter gate;
- generated reports are local evidence and normally remain untracked.

## 7. Local Assistant

Install or invoke the local CLI using the platform-specific scripts already in
the repository. Before use:

- keep authority surfaces loopback-only;
- use generated tokens rather than tokenless defaults;
- keep teacher and network authority request-local and false unless explicitly
  needed;
- never expose the dashboard, OpenAI-compatible shim, or Hive publicly based
  only on local tests.

The assistant can use VCM, deterministic tools, reports, and registered routes.
Those contributions make the result assisted. Record genuine dogfood outcomes
without assigning learned-model credit.

## 8. Data And Teacher Reconstruction

Public source does not include the private frozen corpus. A local operator must
reconstruct data only from approved source manifests and run the full admission
pipeline.

Required dimensions:

- training rights and license;
- provenance and permitted use;
- English and supported programming-language scope;
- exact and semantic deduplication;
- public/evaluation contamination;
- privacy;
- retention;
- tokenizer identity;
- synthetic share;
- teacher-row share and optimizer sampling.

Public benchmark prompts, tests, hidden tests, solutions, traces, labels, and
answer templates remain evaluation-only.

Live teacher calls are OpenAI-only, governed training pressure. They are never
runtime serving.

## 9. Private Checkpoint Restoration

If the private checkpoint is available, validate it through the registered
lineage rather than copying weights alone.

Required state:

- model;
- optimizer;
- MLX RNG;
- data cursor;
- training receipt;
- child and host-guard receipts;
- append-only segment manifests;
- training config and plan migration;
- frozen evaluation identity.

The current private reference state is described in Project State. A public
replicator should expect `NOT_AVAILABLE`, not fabricate a substitute and call
it equivalent.

## 10. Training

Read [Real Training Preflight](REAL_TRAINING_PREFLIGHT.md) before any
state-changing run.

Important:

- source checkout is not training authority;
- a historically GREEN freeze can become stale after legitimate source edits;
- the operator hold must remain until every current gate passes and the
  operator explicitly removes it;
- one bounded fresh-process segment comes before a long continuation;
- exact transactional custody is mandatory;
- capability evaluation remains unconsumed until the candidate and matched
  controls complete.

Do not use old Code-LM, overnight, KERC, ANE, or autonomous-weeks commands as a
shortcut around the current campaign.

## 11. Hive And Distributed Work

Hive is a registered bounded-task runtime, not a remote shell.

Defaults:

- loopback/local;
- signed discovery;
- separate coordinator, worker, and discovery credentials;
- header-only credentials;
- task-kind allowlist;
- fail-closed remote execution without sandbox qualification.

A real multi-node claim requires a trusted reachable peer and a bounded task
receipt. Dry-run plans, stale peers, or synthetic fixtures do not count.

## 12. What A Successful Replication Proves

A public-safe replication can prove:

- source builds;
- deterministic tests pass;
- governance schemas and gates execute;
- Rust reference components behave as tested;
- the public evidence capsule exactly resumes;
- local authority defaults fail closed.

It cannot prove:

- access to the private corpus or checkpoint;
- useful learned capability;
- MoECOT superiority;
- private evaluation results;
- fastest-possible hardware performance;
- LAN/internet security;
- long-run reproducibility without private custody artifacts.

## 13. Troubleshooting Order

1. Check `git status`.
2. Confirm Python/Rust versions.
3. Read the failing gate’s structured faults.
4. Verify referenced source and report identities.
5. Distinguish missing private artifacts from source bugs.
6. Fix the canonical owner.
7. Do not lower gates or create a parallel report family.
8. Preserve the failure as scoped evidence.
