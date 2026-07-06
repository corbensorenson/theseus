# Online Source Catalog

The online source catalog is the governed intake layer for external RL
environments, benchmark frameworks, and training-data candidates.

It is deliberately not a bulk downloader. Its job is to stage auditable source
archives or metadata under ignored folders, then require adapter, license,
leakage, and quality gates before anything becomes training data or an active
benchmark.

## Files

- `configs/online_source_catalog.json`: curated source list, licenses,
  priority, smoke plans, and import policy.
- `scripts/online_source_catalog.py`: validates the catalog and optionally
  stages approved sources.
- `configs/resource_pantry.json`: spillover-drive clone policy for approved
  open-source benchmark/RL/eval repositories.
- `scripts/resource_pantry.py`: creates the shallow-clone pantry and writes
  adapter-readiness status.
- `reports/online_source_catalog_report.json`: machine-readable catalog status.
- `reports/resource_pantry.json`: local clone/metadata/readiness status.
- `D:/ProjectTheseus/resource_pantry/external_benchmark_candidates/`: canonical
  ignored staging area for source archives and dataset metadata on this
  workstation.
- `data/external_benchmark_candidates/`: legacy/small ignored local staging
  area used by a few smoke paths and native sample manifests.

## Current Source Classes

The catalog is split into three practical classes:

- RL/source-code benchmark candidates: Gymnasium, Minigrid, bsuite, Craftax,
  Procgen, PettingZoo, Meta-World, Jumanji, Brax, dm_control, envpool,
  Crafter, and metadata-only gated arcade/text RL references.
- Emulator RL candidates: PyGBA, Gymboy, Stable-Retro, and PyBoy metadata,
  gated on user-supplied ROMs and wrapper smoke tests.
- Language benchmark framework candidates: `lm-evaluation-harness`, HELM,
  BIG-bench, simple-evals, HumanEval, EvalPlus, LiveCodeBench, BigCodeBench,
  MBPP, MMLU-Pro, GPQA, and the queued official BabyLM evaluation pipeline.
- Agentic benchmark candidates: BFCL, tau-bench, tau2-bench, WebArena,
  BrowserGym, OSWorld, TheAgentCompany, GAIA, SWE-bench, mini-SWE-agent,
  SWE-agent, OpenCode, OpenHands, Terminal-Bench, CodeClash, SWE-Atlas,
  SWE-PolyBench, SWE-ReX, SWE-smith, SWE-gen, SWE-Skills-Bench, and
  permissive-source candidates only. ToolSandbox, GitTaskBench, Aider Polyglot,
  OpenCode Bench, SWELancer, and other unclear or queue-only sources are
  excluded from the active runway until permissive license/terms are explicit.
- Voice benchmark/data candidates: SpeechBrain benchmark metadata plus Common
  Voice, LibriSpeech, LibriTTS, LJSpeech, and queued VCTK metadata for the
  Theseus-native voice I/O lane. These sources are pressure/data only;
  pretrained or installed speech inference does not count as system capability.
- Training-data metadata candidates: FineWeb-Edu, FineWeb, FineWeb-2,
  SmolLM Corpus, Dolma, Cosmopedia, APPS, CodeContests, OpenOrca,
  NuminaMath-CoT, Hermes function calling, Synth-APIGen, OpenAssistant,
  The Stack v2, StarCoderData, CodeSearchNet, and queued/license-blocked math
  or language-modeling corpora.

Training corpora stay metadata-only until a separate sampling plan passes
dedupe, leakage, provenance, quality, and ratio gates. The native voice lane has
a narrower exception: `scripts/native_voice_training_manifest.py` may
materialize tiny capped LibriSpeech audio/transcript shards under ignored
storage when `configs/native_voice_training_policy.json` allows it.

Current training-data focused catalog/pantry sweep, 2026-05-22:

```text
catalog training-data sources: 18
approved for catalog import: 13
blocked or queued: 0
excluded non-permissive or uncleared: 5
training-data candidates: 18
resource-pantry source repos on D:: 55
benchmark dataset groups on D:: 3
benchmark dataset bytes on D:: 1,230,528,518
private training rows summarized by runway: 9,135
STS rows summarized by runway: 2,967
governed small-sample rows summarized by runway: 690
adapter cards: 18
smoke-passed cards: 8
local RL environments in registry: 4
```

The open conversation pantry is now part of the near-term focus. It stages
governed tiny samples from allowlisted conversation sources under
`D:/ProjectTheseus/training_data/open_conversation_pantry` and currently
provides 3,043 conversation samples, 2,931 private SFT rows, and 2,931 STS rows.
These rows are private pressure only: no public benchmark solutions, no teacher
distillation, no promotion evidence, and public code-eval overlap tokens are
rejected.

The governed web/instruction sampler writes to
`D:/ProjectTheseus/training_data/governed_samples` and currently provides 178
allowlisted tiny rows plus 512 local pairwise rows. The sampler admits only
configured, license-checked sources and never bulk-downloads.

For coding growth, the catalog now prefers license-clear benchmark/framework
pressure over forcing uncertain sources through the gate. OpenCode full smoke
is unblocked by a project-local Bun toolchain in ignored storage. OpenHands,
Terminal-Bench, SWE-ReX, and SWE-smith are staged and source-contract runnable,
but full sandbox execution still requires Docker or Podman on the host.

For tool-use and agentic work, the new ready runway is BFCL for function/tool
calling, tau-bench and tau2-bench for multi-turn policy/tool dialogues,
BrowserGym/WebArena for web-agent control, OSWorld and TheAgentCompany for
desktop/computer-use pressure, plus SWE-Skills-Bench and SWE-PolyBench for
repo-level coding-agent skills. ToolSandbox, GitTaskBench, and any other
uncleared or queue-only source are excluded from the active runway; they are
not training data, not benchmark pressure, and not blockers until a permissive
license/terms audit promotes them.

## Refresh

Validate the catalog without network fetches:

```powershell
py -3.13 scripts\online_source_catalog.py --out reports\online_source_catalog_report.json
```

Stage approved source archives and metadata:

```powershell
py -3.13 scripts\online_source_catalog.py --allow-network-fetch --import-sources --max-imports 12 --out reports\online_source_catalog_report.json
```

Stage approved open-source repositories into the local resource pantry:

```powershell
py -3.13 scripts\resource_pantry.py --execute --max-clones 24 --out reports\resource_pantry.json --markdown-out reports\resource_pantry.md
```

Stage or refresh the focused coding-agent/tool-use runway:

```powershell
python scripts\resource_pantry.py --execute --max-clones 8 --source-id bfcl --source-id tau2_bench --source-id browsergym --source-id the_agent_company --source-id swe_skills_bench --source-id terminal_bench --source-id swe_polybench --out reports\resource_pantry_agentic_setup.json --markdown-out reports\resource_pantry_agentic_setup.md
```

Refresh the governed training and benchmark runway, then write the consolidated
runway report:

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

The dashboard exposes the same flow in the **Online Source Catalog** panel.

## Safety Rules

- Unknown licenses are queue-only.
- Source archives are staged, not imported into the active benchmark ledger.
- Training-data candidates are metadata-only by default.
- Commercial ROMs and copyrighted game assets are forbidden unless explicit
  user rights and provenance are recorded.
- User-supplied local ROMs are discovered through the local ROM registry, not
  through autonomous downloads.
- Every staged source needs a smoke adapter and benchmark card before it can
  pressure the ratchet.
- Every training sample needs source/version/hash provenance before use.

## Next Integration Step

After staging a source, create a small adapter smoke report before promotion:

```text
source archive/metadata
  -> license and provenance check
  -> minimal install or adapter smoke
  -> benchmark card
  -> contamination/leakage check
  -> benchmark ledger as diagnostic/frontier/public calibration
```

This keeps external discovery useful without letting the system silently turn
the internet into ungoverned training data.
