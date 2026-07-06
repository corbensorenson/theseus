# Data And Artifacts

This repository is intended to publish the project source, configuration,
documentation, small first-party fixtures, and reproducible scripts.

The Apache-2.0 license applies to first-party project code and documentation.
It does not automatically apply to third-party datasets, benchmark clones,
model weights, generated checkpoints, generated reports, or cached artifacts.

## Tracked By Default

- Rust crates, Python scripts, dashboard files, tests, examples, adapters, and
  configuration.
- Documentation under `docs/`.
- Small benchmark fixtures under `benchmarks/`.
- Small local JSON/JSONL data files that are part of the current reproducible
  research state.
- User-supplied personality source documents under `personality-documents/`
  when the user intentionally adds them to the project.

## Ignored By Default

- `reports/`: generated ledgers, benchmark reports, teacher queues, dashboard
  job logs, and runtime telemetry.
- `checkpoints/`: generated major/minor checkpoint chains and materialized
  workspaces.
- `target/`: Rust build output.
- `.venv*/`, `tmp/`, `__pycache__/`: local runtime environments and caches.
- `data/.cache/`: local preprocessing cache.
- `data/synthetic/`: generated synthetic training blends and dataset cards.
- `data/local_roms/`: user-supplied private ROM files for local RL only.
- `data/rom_manifests/`: optional local-only ROM manifests and compatibility
  notes.
- `data/public_benchmarks/`: third-party benchmark clones.
- `data/external_benchmark_candidates/`: queued or fetched external sources
  that still require audit.
- `vendor/pufferlib/`: vendored third-party source copy.
- large model artifacts such as `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`,
  `*.onnx`, and `*.bin`.
- `configs/*.local.json`: machine-specific secrets and local-only operating
  config, including `configs/hive_storage.local.json` for explicit Hive storage
  share roots.

Hive storage pulls are runtime artifacts under `reports/hive_storage_inbox/`.
The source files remain on the owning machine or NAS; only files intentionally
pulled through the authenticated Hive storage API are copied locally.

Large runtime paths should be redirected through `configs/runtime_paths.json`
and `scripts/runtime_paths.py` rather than committed to the source checkout.
On Windows workstations with a secondary data drive, `D:\ProjectTheseus` is a
supported local runtime root for corpora, caches, generated reports,
checkpoints, and build products. On macOS/Linux, use a machine-local path with
the same logical roles and keep it ignored by git.

For space-constrained Windows nodes, the migration helper can move the source
checkout and create a compatibility junction for old shortcuts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\migrate_project_home_to_d.ps1
```

Run it first as a dry run. When services/editors are closed, use `-Execute` and
`-CreateCompatibilityJunction`. The default execution keeps a timestamped
`New project.pre-d-migration-*` backup on C:. Delete that backup manually only
after the D: checkout has passed `git status`, `theseus status`, and a smoke
command. Use `-RemoveBackupAfterJunction` during the first execution only if
you intentionally want the helper to remove that backup immediately after the
junction is created.

## Before Adding New Data

1. Confirm the source license and redistribution terms.
2. Record source, version, hash, and intended role in the relevant registry.
3. Keep public benchmarks as calibration unless they are explicitly promoted by
   the benchmark lifecycle.
4. Keep commercial ROMs or copyrighted game assets out of the repository unless
   explicit rights are documented.
5. Prefer scripts that recreate data from approved sources over committing large
   raw artifacts.

## Online Source Catalog

Curated online RL environments, benchmark frameworks, and training-data
candidates are recorded in:

- `configs/online_source_catalog.json`
- `scripts/online_source_catalog.py`
- `reports/online_source_catalog_report.json`

Approved source archives and metadata are staged only under
`D:\ProjectTheseus\resource_pantry\external_benchmark_candidates\`, which is
outside the tracked checkout. Small legacy or repo-local smoke artifacts may
still appear under ignored `data/external_benchmark_candidates/`, but D: is the
canonical storage root on this workstation. This makes it safe to populate
local benchmark candidates without publishing third-party archives or
accidentally training on unaudited data.

The current policy is:

- code benchmark sources may be staged when the SPDX license is allowlisted;
- training corpora are metadata-only until a sampling plan is approved, except
  tiny explicitly capped governed samples written to
  `D:\ProjectTheseus\training_data\governed_samples` and tiny native voice
  shards allowed by `configs/native_voice_training_policy.json`;
- unknown-license sources are queue-only;
- commercial ROMs or copyrighted game assets are blocked without explicit user
  rights;
- staged sources still need adapter smoke tests, benchmark cards, leakage
  checks, and ledger promotion before use.

## Training Resource Runway

The consolidated runway report is:

- `scripts/training_resource_runway.py`
- `reports/training_resource_runway.json`
- `reports/training_resource_runway.md`

It is a read-only summary over the governed source pantry, D: training shelves,
public benchmark staging reports, and data inventory. It does not download new
data. It exists so the autonomy loop can answer, in one place, what is ready
for training pressure, what is public calibration only, and which gates are
still blocking use.

Current runway snapshot, 2026-05-22:

```text
D:\ProjectTheseus resource source repos: 55
benchmark dataset groups: BigCodeBench, EvalPlus, LiveCodeBench
benchmark dataset bytes: 1,230,528,518
private training rows: 9,135
STS rows: 2,967
governed small-sample rows: 690
counted public calibration tasks: 755
benchmark adapter cards: 18
smoke-passed adapter cards: 8
local RL environments: 4
catalog approved/imported sources across runway: 77
training-data catalog approved/imported sources: 13
training-data catalog blocked or queued sources: 0
training-data catalog excluded as non-permissive or uncleared: 5
runway state: YELLOW
```

Public benchmark assets are explicitly eval/calibration-only. The staged
EvalPlus, BigCodeBench, and LiveCodeBench payloads must not be admitted into
private training rows. Private pressure comes from locally generated residual
code rows, private repo-repair rows, permissive open-code expression rows,
governed open-conversation rows, and tiny overlap-checked web samples.

The conversation lane is temporarily prioritized so Theseus can become
directly talkable in English before the next code-transfer push. Current
governed open-conversation pantry status:

```text
conversation samples: 3,043
private SFT rows: 2,931
STS rows: 2,931
bulk download: false
promotion evidence: false
public benchmark solution overlap: rejected
teacher distillation: false
```

The latest governed sample pass also materialized 178 tiny allowlisted
Hugging Face rows and 512 local pairwise rows under
`D:\ProjectTheseus\training_data\governed_samples`. Admitted sources include
FineWeb-Edu, SmolLM Corpus, OpenOrca, NuminaMath-CoT, Hermes function calling,
and Synth-APIGen. Gated, unclear-license, or terms-blocked sources remain
metadata-only or excluded.

Refresh and summarize the runway with:

```powershell
python scripts\stage_evalplus_public_data.py --out reports\stage_evalplus_public_data.json
python scripts\stage_public_code_benchmark_data.py --out reports\public_code_benchmark_data_stage.json --live-shards 3
python scripts\open_code_training_pantry.py --repo-config configs\open_code_training_pantry_expanded.json --root D:/ProjectTheseus/training_data/open_code_pantry --refresh --out reports\open_code_training_pantry.json
python scripts\open_conversation_training_pantry.py --config configs\open_conversation_training_pantry.json --root D:/ProjectTheseus/training_data/open_conversation_pantry --allow-network-fetch --refresh --out reports\open_conversation_training_pantry.json --markdown-out reports\open_conversation_training_pantry.md
python scripts\training_data_sampler.py --policy configs\autonomy_policy.json --catalog configs\online_source_catalog.json --catalog-report reports\online_source_catalog_report.json --allow-network-fetch --sample-root D:/ProjectTheseus/training_data/governed_samples --out reports\training_data_sampler.json
python scripts\benchmark_adapter_factory.py --write-cards --out reports\benchmark_adapter_factory.json --markdown-out reports\benchmark_adapter_factory.md
python scripts\rl_benchmark_registry.py --refresh-local --out reports\rl_benchmark_registry.json
python scripts\training_data_inventory.py --out reports\training_data_inventory.json
python scripts\training_resource_runway.py --out reports\training_resource_runway.json --markdown-out reports\training_resource_runway.md
```

## Native Voice Data

Native speech input/output is trained as part of the Theseus head/router I/O
boundary. The governed source of truth is:

- `configs/native_voice_policy.json`
- `configs/native_voice_training_policy.json`
- `scripts/native_voice_training_manifest.py`
- `reports/native_voice_training_manifest.json`

The manifest currently tracks LibriSpeech, LibriTTS, LJSpeech, Common Voice,
and VCTK for STT/TTS pressure. It may automatically materialize tiny
LibriSpeech audio/transcript shards under
`data/external_benchmark_candidates/native_voice_samples/`, which is ignored by
git. `scripts/native_voice_bootstrap_learner.py` can then write local bootstrap
component reports and ignored index artifacts from those samples. Bulk speech
archives, uncertain-license shards, and all provider or pretrained STT/TTS
inference remain forbidden without explicit approval.

## Personality Documents

`personality-documents/` is a user-owned source area for writings, exported
tweets, and other personal reference material. It is not treated as generic
training data by default.

The governed source of truth is:

- `configs/personality_core_policy.json`
- `configs/personality_drift_eval.json`
- `configs/belief_update_policy.json`
- `scripts/personality_core.py`
- `scripts/personality_context_builder.py`
- `scripts/personality_drift_eval.py`
- `scripts/belief_update_governor.py`
- `reports/personality_core.json`
- `reports/personality_context_last.json`
- `reports/personality_drift_eval.json`
- `reports/belief_update_governance.json`
- `reports/belief_update_ledger.jsonl`
- `reports/personality_core_training_manifest.jsonl`

The default policy includes markdown/text documents and selected public-writing
files from Twitter/X archives (`tweets.js`, `note-tweet.js`, `article.js`).
It excludes direct messages, ads, IP/device/account metadata, likes, follows,
Grok chats, and archive assets. Generated personality reports stay under
ignored `reports/`; adapter training or public export requires an explicit
user-approved step. Observation-to-belief updates are ledgered locally and can
be accepted, sent for review, or quarantined before changing durable system
orientation.

## Local ROM Assets

The system can inventory ROMs that the user supplies locally, but it must not
download ROMs or commit ROM contents. The relevant files are:

- `configs/local_rom_policy.json`
- `scripts/local_rom_registry.py`
- `reports/local_rom_registry.json`

Use `data/local_roms/`, `SPARKSTREAM_ROM_ROOTS`, or the `--rom-root` argument
to point the registry at private ROMs. External absolute paths are redacted in
reports; hashes and display names are retained so wrapper smoke tests and
regression reports can be reproduced locally.

## Synthetic Data

Generated synthetic data is a runtime artifact. The source of truth is the
curation script, policy, report, and dataset card:

- `scripts/synthetic_data_curator.py`
- `configs/synthetic_data_policy.json`
- `reports/synthetic_data_curator.json`
- `data/synthetic/*.dataset_card.json`

Synthetic rows must remain provenance-tagged, leakage-checked against eval and
holdout files, capped by ratio, and validated through public/private/regression
candidate gates before any promotion claim.

## Knowledge Sources

Knowledge sources are not automatically training datasets. A site can be useful
for targeted lookup, claim checking, topic discovery, or benchmark ideas while
still being blocked from autonomous bulk ingestion.

Grokipedia is registered as a lookup-only candidate in
`configs/external_benchmarks.toml` and `configs/autonomy_policy.json`. Do not
bulk scrape it, train on copied pages, or distill model/service output from it
until the terms, robots posture, per-page or dataset license, provenance, and
human-approval gates have all passed.
