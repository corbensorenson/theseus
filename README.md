# Project Theseus

Project Theseus is a private, locally trained AI research system built around
specialist neural arms, governed memory and tools, exact experiment custody,
and zero external inference at serving time.

The project asks one practical, falsifiable question:

> Can a locally trained Theseus student do useful conversation and repository
> work, and does modular MoECOT improve the useful-safe frontier over matched
> dense controls at acceptable total cost?

Theseus is not yet a demonstrated useful learned assistant or a production
runtime. Its governance, evidence, and orchestration infrastructure are more
mature than its measured learned capability.

## Current State

The canonical human-readable status is
[Project State](docs/PROJECT_STATE.md). The short version as of 2026-07-29 is:

- the exact 57.3M-parameter modular checkpoint is paused at optimizer step
  11,416 with model, AdamW, MLX RNG, cursor, and prospective lineage custody;
- the operator hold is installed, so long training is `TRAINING_HELD`;
- the frozen 160-case private functional evaluation is unconsumed and the
  checkpoint remains `NOT_EVALUATED`;
- both matched dense controls remain untrained;
- the selected practical route is compiled FP32 MLX;
- KERC/RDC is a protected, frozen successor experiment and ANE training was
  not selected for the current campaign;
- local security and evaluator-integrity repairs pass their bounded tests, but
  LAN or public exposure is not qualified;
- teacher-accepted rows are about 0.013% of accepted training rows;
- genuine daily usefulness remains `EMPIRICAL_SUPPORT_INSUFFICIENT`.

Documentation consolidation does not change any of those evidence states.
The next actions are in the [roadmap](roadmap.md).

## Operating Boundaries

The complete charter is in [AGENTS.md](AGENTS.md). Its core constraints are:

1. External inference may be used only as a governed training teacher.
   Externally generated tokens are never served.
2. Public benchmarks are calibration-only and never become training data.
3. Candidate generation and ranking cannot see hidden answers, tests,
   answer-family labels, or fields derived from them.
4. Templates, tools, retrieval, routers, and deterministic fallbacks receive
   no learned-generation credit.
5. Negative evidence is scoped to the exact implementation and regime tested.
6. English, Python, JavaScript/TypeScript, HTML/CSS, and Rust are the current
   seed scope.
7. New lanes and surfaces remain frozen unless they directly serve the neural
   seed, teacher accounting, real dogfood, or repository health.

## System Shape

Theseus has five distinct planes:

| Plane | Role | Capability implication |
| --- | --- | --- |
| Neural experiment | Modular MoECOT student and matched dense controls | Only blind frozen evaluation can establish learned capability |
| Protected discovery | SymLiquid/CGS/VSA/liquid substrate and bounded successor mechanisms | Research hypothesis, not the practical route |
| Local product | CLI, VCM, tools, assisted workflows, and personality context | Useful mechanics; assisted outcomes are reported separately |
| Governance | Registries, ledgers, evaluator integrity, data/teacher gates, and exact custody | Makes claims auditable; does not create capability |
| Runtime infrastructure | SparkStream, Hive, dashboard, device adapters, and update machinery | Local prototype infrastructure; exposure needs separate qualification |

The active neural experiment uses a shared transformer trunk with independent
English, Python, JavaScript/TypeScript, HTML/CSS, and Rust arms behind the
Octopus/MoECOT route contract. It is compared with:

- a dense control matched on active parameters; and
- a dense control matched on total parameters.

The older Rust-first SymLiquid stack remains a protected discovery comparator.
Its compact loop is:

```text
observe -> compress -> expand -> bind -> predict -> act -> correct -> recompress
```

That prototype must not be confused with the active 57.3M MLX campaign.

## Repository Map

| Path | Purpose |
| --- | --- |
| `configs/project_manifest_registry.json` | Canonical implementation and route ownership |
| `configs/roadmap_implementation_matrix.json` | Detailed remaining obligations |
| `docs/PROJECT_STATE.md` | Current human-readable state |
| `roadmap.md` | Compact execution order |
| `docs/README.md` | Documentation index and authority classes |
| `docs/GLOSSARY.md` | Shared terminology and evidence-state meanings |
| `scripts/` | Experiment, governance, evaluation, and operator entry points |
| `tests/` | Python regression and integrity coverage |
| `crates/` | Rust prototype and runtime components |
| `apps/` | Local UI and operator surfaces |
| `reports/` | Generated evidence; not progress by itself |
| `runtime/` | Private local state and ledgers |
| `checkpoints/` | Private model and optimizer state |

Generated reports, private traces, datasets, credentials, and checkpoints do
not belong in the public tracked source tree.

## Start Safely

Use Python 3.12+ and a current Rust toolchain. Platform-specific MLX work
requires Apple Silicon and the pinned local environment described in the
[Replication Guide](docs/REPLICATION_GUIDE.md).

Run the lightweight source checks:

```bash
python3 scripts/theseus_doc_link_audit.py
python3 scripts/public_release_audit.py --gate
python3 scripts/theseus_project_registry.py --gate
python3 scripts/roadmap_implementation_gate.py --gate
python3 -m pytest -q
cargo test --workspace
```

Some suites are intentionally platform- or artifact-gated. A skipped private,
MLX, Metal, or checkpoint test is not equivalent to a pass.

For local assisted use:

```bash
python3 scripts/theseus_local_assistant.py --help
```

Assisted outputs may use deterministic tools, memory, retrieval, or rules.
Record them as assisted and never treat them as learned-model capability.

## Training

Do not infer launch authority from a command in this repository. Read
[Real Training Preflight](docs/REAL_TRAINING_PREFLIGHT.md) and current
[Project State](docs/PROJECT_STATE.md) first.

At present:

- no long run is authorized;
- the source-bound readiness package must be refreshed after the documentation
  transaction;
- the operator hold remains installed;
- the frozen evaluation must remain untouched;
- no additional KERC, ANE, optimizer, architecture, data, or generic
  acceleration experiment should interrupt the selected matched campaign.

A future campaign action requires an exact fresh-process qualification, current
gates, transactional checkpoint headroom, and explicit operator approval.

## Documentation

Read documentation in this order:

1. [Project State](docs/PROJECT_STATE.md)
2. [Roadmap](roadmap.md)
3. [Glossary](docs/GLOSSARY.md)
4. [Documentation Index](docs/README.md)
5. [Top-To-Bottom Architecture](docs/TOP_TO_BOTTOM_ARCHITECTURE.md)
6. [Replication Guide](docs/REPLICATION_GUIDE.md)

Current state belongs only in Project State. Detailed implementation ownership
belongs in the project registry, detailed obligations in the roadmap matrix,
and observations in content-bound reports and ledgers. Architecture and
research documents do not override those sources.

## Public Source And License

The intended public source repository is
[corbensorenson/theseus](https://github.com/corbensorenson/theseus). Before a
public snapshot, run the public-release and registry gates and inspect the
actual Git diff.

See [LICENSE](LICENSE), [Data And Artifacts](docs/DATA_AND_ARTIFACTS.md), and
[Public Release](docs/PUBLIC_RELEASE.md) for source, data, model, and generated
artifact boundaries.

## Honest Limitations

- There is no capability claim for the current neural checkpoint.
- The matched modular-versus-dense experiment is incomplete.
- The first explicit travel-mode dogfood task was recorded as a miss.
- Hosted CI has not yet verified the latest local workflow changes.
- Local security tests do not qualify Theseus for LAN or public exposure.
- The complete predecessor checkpoint chain before the prospective step-9,048
  lineage was not retained.
- The protected discovery architecture has not been shown to outperform the
  practical transformer route.

These are the project’s current walls, not documentation defects to be worded
away.
