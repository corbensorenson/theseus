# ROM, RL, Data, and Benchmark Growth Lanes

Last updated: 2026-05-11.

This document describes the governed expansion lanes that let SparkStream/RMI
grow beyond the current BabyLM and Ocean-style local tasks without turning into
an unbounded downloader.

## Policy

- ROMs are user-supplied private assets only. The system must never download
  commercial ROMs.
- ROM files, save states, and emulator caches stay out of git.
- External benchmark and dataset sources are staged as metadata or source
  archives first, then promoted only after license, adapter, smoke, leakage,
  and resource checks.
- External/API/local-model inference is only allowed through the sparse teacher
  role. Data discovery and benchmark metadata fetches are not inference, but
  they are still governed by source policy.
- A saturated benchmark should become regression. Remaining failures go to
  residual escrow. New pressure should come from harder frontiers, bridge tasks,
  RL environments, or teacher-proposed architecture changes.

## Local ROM Inventory

The local ROM registry lives in:

- `configs/local_rom_policy.json`
- `scripts/stage_local_rom_assets.py`
- `scripts/local_rom_registry.py`
- `reports/local_rom_staging_report.json`
- `reports/game_asset_inventory.json`
- `reports/local_rom_registry.json`

Ways to expose your private collection:

```powershell
# Option 1: copy or symlink private files under an ignored project folder
mkdir data\local_roms

# Option 2: leave the collection elsewhere for this shell/session
$env:SPARKSTREAM_ROM_ROOTS = "D:\Your\Private\ROM\Collection"

# Option 3: one-shot scan
python scripts\local_rom_registry.py --rom-root "D:\Your\Private\ROM\Collection"
```

If the collection is placed in the project-root `games/` inbox, stage active
GB/GBC/GBA assets first:

```powershell
python scripts\stage_local_rom_assets.py --source-root games --execute --out reports\local_rom_staging_report.json
```

The staging tool copies or extracts only currently supported GB/GBC/GBA assets
into `data/local_roms/<system>/`, deduplicates by SHA-256, and inventories
inactive NDS/N64/disc assets in `reports/game_asset_inventory.json`. It leaves
the original `games/` collection untouched. `.zip` is supported through the
Python standard library; `.7z` uses optional `py7zr` when installed and
otherwise falls back to Windows `tar` when possible.

The report stores hashes, display names, sizes, system type, and redacted paths
for external roots. It does not store ROM contents.

## ROMs Worth Prioritizing

Start with ROMs that already map cleanly onto available wrappers or reward
designs:

| Priority | System | ROM family | Why |
| --- | --- | --- | --- |
| High | GBA | Pokemon Emerald | PyGBA has a Pokemon Emerald reward wrapper, making this the fastest GBA shaped-RL path. |
| High | GB/GBC | Tetris | Compact state, score-based rewards, planning, reflex, and long-horizon control. |
| High | GB/GBC | Pokemon Red, Blue, Yellow | Useful for navigation, memory, anti-loop behavior, and curriculum tasks via Gymboy/PyBoy-style wrappers. |
| Medium | GB/GBC | Kirby, Super Mario Land, Wario, Metroid | Platform/control pressure with sparse rewards and hazards. |
| Medium | GBA | Advance Wars, Fire Emblem, tactics games | Planning and delayed-credit frontiers after custom wrappers exist. |

## Emulator RL Sources

Current catalog entries:

- PyGBA: `dvruette/pygba`, MIT, GBA wrapper around mGBA with Gymnasium support
  and a Pokemon Emerald wrapper.
- Gymboy: PyPI package, MIT, Game Boy/Game Boy Color Gymnasium environments for
  Tetris, Pokemon, Kirby, and Super Mario Land variants.
- Stable-Retro: `Farama-Foundation/stable-retro`, MIT, emulator-backed RL with
  local ROM import workflows.
- PyBoy: useful GB/GBC emulator surface, but currently blocked in the catalog
  until GPL compatibility is explicitly accepted for the local integration.

The autonomy loop can now discover and queue these sources. Promotion requires:

1. local user ROM match;
2. package/import smoke;
3. deterministic reset and step smoke;
4. documented reward and save-state policy;
5. resource profile;
6. benchmark ledger entry with graduation and residual escrow policy.

## Open RL Frontiers

The system already has local Puffer/Ocean-style environments and now keeps
open-source RL source candidates in `configs/online_source_catalog.json`.

High-value growth lanes:

- Gymnasium classic control and toy text for API compatibility.
- Minigrid for partial observability, memory, instruction following, and
  planning.
- bsuite for diagnostic exploration, memory, and credit-assignment metrics.
- Procgen for visual generalization after Windows build/runtime checks.
- Craftax for open-ended survival/crafting once optional JAX costs are profiled.
- PettingZoo for multi-agent pressure.
- Metaworld for robotics only after MuJoCo and Windows support are verified.

## Language Benchmarks and Data

Benchmark/catalog growth lanes:

- lm-evaluation-harness for public language/reasoning calibration once the local
  scoring adapter is explicit.
- HELM for broad scenario and multi-metric calibration.
- BIG-bench as a task source for bridge benchmarks and diagnostics.

Training-data metadata lanes:

- FineWeb-Edu, SmolLM Corpus, Dolma, and Cosmopedia are metadata-first
  candidates.
- Sampling must remain tiny and governed until dedupe, leakage, PII, quality,
  source mix, and synthetic-ratio checks pass.
- OpenWebMath stays blocked until license/data-sheet uncertainty is resolved.

## Dashboard and Autonomy Integration

The dashboard status API includes `local_rom_staging`, `game_asset_inventory`,
and `local_rom_registry`. The autonomy cycle stages local ROM assets, refreshes
the ROM registry, updates RL benchmark recommendations, and queues a local ROM
wrapper lane when matched profiles exist.

Useful commands:

```powershell
python scripts\local_rom_registry.py --out reports\local_rom_registry.json
python scripts\stage_local_rom_assets.py --source-root games --execute --out reports\local_rom_staging_report.json
python scripts\rl_benchmark_registry.py --refresh-local --out reports\rl_benchmark_registry.json
python scripts\online_source_catalog.py --catalog configs\online_source_catalog.json --out reports\online_source_catalog_report.json
```

For long runs, start SparkStream normally. Teacher and governed network
discovery default on through policy, while teacher apply mode and unsafe/bulk
downloads remain blocked:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_sparkstream.ps1 -Restart -StartDaemon -Profile inner_loop -Execute -DurationHours 10 -Port 8787
```
