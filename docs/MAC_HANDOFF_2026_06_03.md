# Theseus Mac Handoff - 2026-06-03

This is the current handoff note for moving Project Theseus / SparkStream from
the Windows CUDA workstation to Apple Silicon Codex.

## Git State

- Canonical local branch: `main`
- Remote: `origin` at `https://github.com/corbensorenson/theseus.git`
- Before starting Mac work, follow the newest Windows-forward remote tip:

  ```bash
  python3 scripts/sync_latest_windows_lane.py
  python3 scripts/sync_latest_windows_lane.py --switch
  ```

  The helper fetches GitHub, ranks `origin/*` by commit date, refuses to switch
  a dirty worktree, and fast-forwards the selected local branch.
- Before the handoff commit/push, local `main` was ahead of `origin/main` by
  `2288` commits and behind by `0`.
- This should be published as a normal fast-forward `git push origin main`.
  Do not force-push unless a later fetch shows remote divergence.

## Current Capability State

- Coherence gate: `GREEN`
- Candidate promotion: `26/28`
- Remaining promotion blockers:
  - `broad_public_code_transfer_ready`
  - `maturity_integrity_audit_green`
- Public calibration: locked
- Model growth: locked
- Candidate promotion: locked
- Latest control-plane next work:
  `broad_public_transfer_floor_private_repair`

## Mac Control-Plane Review - 2026-06-03

Base source tip reviewed on Mac before this hardening pass:

- branch: `main`
- current commit/version: report-driven; use `git log -1`,
  `reports/hive_verified_version.json`, and the latest
  `reports/hive_macos_release_gate*.json` instead of hard-coding this field.

Mac runtime state:

- `theseus runtime doctor` / `python3 scripts/macos_runtime_doctor.py` is
  `GREEN` after Mac runtime root normalization.
- The doctor now compares source-checkout and installed-app verified version,
  update catalog, version status, and update check-in summaries by execution
  context, while only reporting presence/mtime for local license/join files.
- Source venv MLX is available:
  `./.venv-puffer/bin/python`.
- Installed app venv MLX is available:
  `~/Library/Application Support/Project Theseus Hive/app/current/.venv-puffer/bin/python`.
- Source-dev and installed LaunchAgent runtime reports now use the same default
  root: `~/Library/Application Support/Project Theseus Hive/runtime`.
- Treat active-shell MLX failures as false negatives unless the Hive runtime
  doctor also says the source/app venv lacks MLX.

Current training/control-plane state after the June 3 Mac hardening pass:

- The Mac is visible as the Apple MLX worker/operator node.
- The Windows workstation remains registered as the NVIDIA CUDA worker node,
  but live API reachability from this Mac is currently blocked at
  `http://10.0.0.147:8791`.
- `hive_training_orchestrator.py plan --profile smoke` skips Windows with
  `unreachable_api` and only assigns reachable local MLX work.
- `hive_node_registry.py` uses lightweight `/api/hive/status` and
  `/api/hive/peers` for scheduler truth by default; the heavier
  `/api/hive/operator/status` remains an operator UI endpoint and is not a
  scheduler dependency.
- The network doctor now emits compact finding-code lists plus shared-secret
  probe interpretation. A successful authenticated probe counts as secret
  proof even if an older peer omits `security.shared_secret_configured`.
- The network doctor retries endpoint probes and records retry telemetry.
  Retry-recovered or registry-flapping coordinators/peers remain RED for
  distributed training readiness until stable probes repeat without recovery.
- `hive_node_registry.py` embeds the latest network doctor state and exposes
  `distributed_training_ready`, `mixed_cuda_mlx_training_ready`, and
  `remote_cuda_live_ready` in the summary.
- `hive_training_orchestrator.py` writes `network_doctor` and
  `fleet_readiness` into its report/status path, so mobile/operator surfaces
  can show `coordinator_unreachable` / `peer_inbound_only_outbound_blocked`
  instead of only saying a CUDA arm had no visible slot.

Live bounded-work proof from Mac:

- Local Mac MLX reports completed for:
  `mac_local_mlx_eval_goal`, `mac_local_mlx_train_goal`, and
  `mac_local_mlx_rollout_goal`.
- Windows CUDA tasks queued from Mac and completed with `returncode=0` for:
  `mac_to_windows_cuda_eval_goal`, `mac_to_windows_cuda_train_goal`, and
  `mac_to_windows_cuda_rollout_goal`.
- Direct artifact fetch from Windows succeeded for those CUDA worker reports and
  saved them under `reports/hive_artifact_inbox/windows_goal_direct/`.
- Indexed artifact sync to the old Windows service still times out. The source
  now includes a bounded recent-first artifact indexer; Windows must receive this
  update before normal indexed sync is considered fixed.

Release/update proof after Mac hardening commit:

- commit: `60c49598d590e0894449f7840a425661496db706`
- verified Hive version: `hive-b63088d7b72a13a7`
- soft update id installed by convergence: `theseus-hive-9902c164ae0627e3`
- macOS release gate: `ok=true`, `private_canary_ready=true`,
  `public_distribution_ready=false`, `fleet_rollout_ready=false`
- local and Windows nodes both reported `status=current` after soft convergence;
  no hard source/app replacement was applied.
- rebuilt artifacts:
  - `dist/macos/ProjectTheseusHive.dmg`
  - `dist/macos/ProjectTheseusHive.pkg`
  - `dist/macos/ProjectTheseusHive.zip`

Release readiness is not green yet:

- the physical Intel Mac canary has not run yet;
- Developer ID signing/notarization is still pending for easy public installs.

MLX parity state:

- Hive worker parity exists for `mlx_eval_chunk`, `mlx_training_chunk`, and
  `mlx_rollout_chunk`.
- `symliquid-cli` now also exposes the Mac command surface
  `train-standalone-mlx`, `train-rollout-mlx`,
  `train-rollout-mlx-sweep`, and `train-token-superposition-mlx`. These run
  bounded first-party MLX bridges through `scripts/macos_mlx_training.py`.
- The audit remains `YELLOW` because the deeper Rust/Metal kernel ports are
  still pending. Do not claim native Rust hot-loop parity until those ports
  replace the Python MLX bridge for the CUDA hot paths.
- `reports/macos_mlx_parity_audit.json` now includes `routing_decisions`:
  registered MLX chunks route to Apple Silicon, runnable MLX CLI bridges are
  valid bounded Mac work, and CUDA-equivalent hot-loop proof still routes to
  Windows/NVIDIA until the Rust/Metal or Rust/MLX port is ready.

Next Mac goal:

```text
Make macOS a first-class Theseus Hive lane: every Mac gate must use the correct
Hive runtime, Apple Silicon must execute MLX eval/training/rollout chunks, Intel
Macs must install cleanly as CPU/storage/operator nodes, the Mac must not queue
Windows CUDA work while the Windows API is unreachable, Windows CUDA and Mac MLX
must pass a fresh live two-node execute proof after reachability is green, and
the rebuilt DMG/pkg/update catalog must install and soft-update spare Macs
without Codex.
```

Current Mac gate entry points:

```bash
theseus mac roles --write-local-config
theseus mac training-preflight --execute --offline --allow-battery-smoke
theseus mac dmg-readiness --execute
theseus mac join-bundle
theseus mac app-status --text
theseus mac canary --execute --write-join-bundle
```

`theseus mac training-preflight` is the required local training gate before
starting long Mac work. It rejects bad runtime/disk/thermal/battery state,
proves Apple Silicon MLX by queueing a local `mlx_training_chunk` and waiting
for the worker report, and keeps offline mode free of artifact sync, teacher
use, and external inference. Intel Macs pass as CPU/storage/operator nodes only
and must not advertise MLX.

`theseus mac dmg-readiness --execute` rebuilds the app/pkg/zip/dmg, publishes
the current verified Hive catalog, refreshes installer artifacts, and checks
that the installed app can serve update catalog and installer artifacts without
Codex. Token-bearing one-click Mac join profiles are generated by
`theseus mac join-bundle` and can be embedded in a private DMG with
`THESEUS_MACOS_JOIN_BUNDLE=dist/macos/ProjectTheseusHive.join.json`.
The bundled profile should include Bonjour `.local` Mac endpoints, private IP
fallbacks, update catalog URLs, installer artifact URLs, and the roaming
handoff policy used by the native iPhone/Watch apps. Treat the generated join
JSON and QR as private token-bearing files.

## Latest Private Repair Evidence

The latest private-only Code LM train-once/fanout completed `GREEN`:

- private candidate rows: `373`
- public candidate rows: `0`
- private candidate promotion-ready rate: `0.941019`
- public task/candidate sidecars: intentionally empty
- external inference calls: `0`
- public calibration unlock semantics: explicitly false

Important source fixes in this handoff:

- `scripts/candidate_promotion_gate.py`
  - coding frontiers no longer double-count generic ARM transfer artifacts;
    they use `code_frontier_transfer_artifact_ready` and
    `code_frontier_transfer_consumed`.
- `scripts/code_lm_train_once_fanout.py`
  - full `--private-only` runs now use explicit empty public sidecar paths.
- `scripts/theseus_control_plane.py`
  - control/lease processes no longer self-count as active Code LM workers;
    locked public-calibration boundaries route to non-public repair work.

## First Commands On Mac

Run these before heavy work:

```bash
python3 scripts/theseus_cli.py runtime doctor --out reports/macos_runtime_doctor.json
python3 scripts/macos_dependency_bootstrap.py --require-mlx --out reports/macos_dependency_bootstrap.json
python3 scripts/macos_mlx_parity_audit.py --out reports/macos_mlx_parity_audit.json --markdown-out reports/macos_mlx_parity_audit.md
python3 scripts/macos_mlx_work_proof.py --out reports/macos_mlx_work_proof.json --markdown-out reports/macos_mlx_work_proof.md
python3 scripts/hive_training_link_doctor.py --refresh --out reports/hive_training_link_doctor.json --markdown-out reports/hive_training_link_doctor.md
python3 scripts/hive_training_orchestrator.py plan --profile smoke --out reports/hive_training_orchestrator.json
python3 scripts/resource_aware_execution_policy.py --out reports/resource_aware_execution_policy.json --markdown-out reports/resource_aware_execution_policy.md
python3 scripts/coherence_delirium_gate.py --out reports/coherence_delirium_gate.json
python3 scripts/candidate_promotion_gate.py
python3 scripts/maturity_integrity_audit.py --out reports/maturity_integrity_audit.json --markdown-out reports/maturity_integrity_audit.md
python3 scripts/asi_wall_breaker_governor.py --out reports/asi_wall_breaker_governor.json --markdown-out reports/asi_wall_breaker_governor.md
python3 scripts/theseus_control_plane.py --out reports/theseus_control_plane.json --markdown-out reports/theseus_control_plane.md
python3 scripts/report_evidence_store.py --out reports/report_evidence_store.json --markdown-out reports/report_evidence_store.md
```

For the macOS Hive app/release lane, run on macOS only:

```bash
python3 scripts/hive_macos_release_gate.py --skip-build --skip-version-publish --skip-local-install --skip-local-converge --out reports/hive_macos_release_gate.json --markdown-out reports/hive_macos_release_gate.md
```

Use `--execute` only when intentionally rebuilding/installing the local Mac
Hive canary.

## Do Not Do These Automatically

- Do not run public calibration without an explicit new operator message.
- Do not train on public benchmark tests, solutions, answers, or
  uncertain-license data.
- Do not grow the model unless the existing model-growth gates allow it.
- Do not treat copied latest JSON reports as durable evidence unless
  `report_evidence_store` is current and GREEN.

## Mac-Specific Notes

- Many Windows reports reference `D:/ProjectTheseus/...` training-data paths.
  On Mac, verify artifact sync or regenerate governed private rows before
  launching heavy training.
- Prefer MLX/Apple-Silicon paths on Mac; do not assume CUDA-specific commands
  are available.
- Treat `reports/macos_mlx_parity_audit.json` as the Mac-native coverage
  ledger. The Hive-level MLX eval/train/control chunks should be available on
  Apple Silicon when MLX is installed. The `symliquid-cli` MLX bridge commands
  are usable now for bounded Mac work, while deeper Rust/Metal rollout and
  token-superposition kernels are still the next implementation target.
- Treat `reports/macos_mlx_work_proof.json` as the runnable Apple-Silicon
  evidence ledger. It runs tiny registered MLX worker chunks plus tiny MLX
  command-bridge smokes and is now part of the macOS release gate.
- Start from the control plane after dependency/bootstrap checks. If the
  control plane still points at `broad_public_transfer_floor_private_repair`,
  continue with private-only transfer repair or the strongest non-public
  blocker it names.
