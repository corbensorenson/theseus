# Project Theseus Hive

Project Theseus Hive is the trusted-device runtime for Project Theseus. It
turns home, workshop, travel, phone, NAS, Raspberry Pi, old Mac, Windows, Linux,
and Apple Silicon machines into one private operator and compute fabric.

This document is the consolidated Hive operations guide. Older long-form notes
have been folded into these current contracts, and this page now covers both
the Mac/operator packaging work and the Windows CUDA/board-executor runtime.

## Current State

Implemented:

- `scripts/hive_operator_os.py` provides the shared operator contract: one
  command vocabulary for dashboard, mobile, CLI, tray/menu bar, relay, and
  future chat-channel adapters; a durable SQLite work board; background tasks;
  persistent goals; Hive Skills; tool hooks; feedback routing; and execution
  safety reports;
- `scripts/hive_work_board_executor.py` consumes that board as the unattended
  work source of truth, assigns tasks to the current best node, runs bounded
  work, writes hook/run/feedback ledgers, retries once, and blocks stale or
  unsupported work with evidence;
- `scripts/high_transfer_curriculum_scheduler.py` keeps benchmark work
  generalized by queuing transferable concept pressure before benchmark-name
  pressure;
- cross-platform Hive node daemon and HTTP API on port `8791`;
- signed LAN peer discovery, Bonjour/DNS-SD advertisement where available,
  durable peer registry, coordinator URLs, and authenticated
  relay/private-tunnel operation;
- Hive Network Doctor for local API, coordinator, firewall, stale-peer, and
  roaming path diagnosis;
- setup wizard, CLI, Windows tray, macOS menu bar app, PWA operator, native iOS
  shell, native Apple Watch companion, native Android shell, and shared spatial
  operator contract for visionOS/Quest clients;
- per-user operator tokens with roles for family/shared Hives;
- bounded registered remote task kinds only, never arbitrary shell;
- storage shares for explicit folders, mounted NAS paths, and peer file
  previews/downloads;
- governed remote-control handoffs for RDP, VNC/macOS Screen Sharing, RustDesk,
  and Sunshine/Moonlight;
- room-aware voice-following presence and listen/respond routing without raw
  audio relay;
- decentralized bounded training orchestration across CPU, CUDA, MLX, and
  Apple Silicon nodes;
- always-busy private utilization sweeps for safe idle slots;
- dry-run-first rented compute/storage planning;
- verified Hive-version catalogs and soft update convergence;
- macOS `.app`, `.zip`, `.pkg`, and unsigned `.dmg` artifacts that target Intel
  and Apple Silicon Macs.
- scheduler placements include internal work-credit gas quotes, and accepted
  worker chunks emit receipts that settle through the compute-market ledger;
- remote task execution is limited to registered task kinds in
  `configs/hive_policy.json`;
- arbitrary shell, teacher calls, git pushes, ROM imports, and bulk data
  downloads are never accepted as remote Hive tasks, and hard app/source
  updates are not remotely forced;
- remote task submission requires `THESEUS_HIVE_SECRET` unless the request is
  loopback.

Important limits:

- Signing/notarization is not configured yet. macOS artifacts are unsigned and
  Gatekeeper may reject them until an Apple signing identity and notarization
  flow is added.
- Phones and watches are operator clients, not training workers.
- Public contribution is still worker-only design/scaffold; private data,
  teacher access, arbitrary filesystem access, ROMs, and raw shell are outside
  public authority.
- A phone cannot reach a private home LAN over cellular without a tunnel or
  relay. Use a self-hosted WireGuard/private tunnel first, or an HTTPS-protected
  Hive relay.
- Background push to iPhone/Apple Watch is not APNs-backed yet. Native apps can
  poll and relay the authenticated Hive notification feed while active; true
  always-on remote notifications require signing plus an APNs or relay bridge.

## Trust Model

Hive has two separate access concepts:

| Access type | Used by | Stored where | What it grants |
| --- | --- | --- | --- |
| Machine join token | Trusted nodes that join the Hive. | `configs/hive_join.local.json` or active Hive profile. | Node-to-node trust and owner-level compatibility. |
| User/operator token | People and phones. | Hashed in `configs/hive_users.local.json`; plaintext only in generated invite output. | Role-scoped operator access. |

Legacy invite tokens still authenticate as `owner` so existing nodes and phone
profiles continue to work. New family/shared setups should create per-user
operator tokens instead of sharing the machine join token everywhere.

Default roles:

| Role | Intended use |
| --- | --- |
| `owner` | Full operator access; legacy Hive join token maps here. |
| `operator` | Trusted adult/admin helper; can view storage, remote-control, and queue bounded eval/training smoke tasks, but cannot manage users or updates. |
| `member` | Normal family user; chat, status, storage browsing/downloads, and voice presence; no remote control or training chunks. |
| `child` | Status, chat, and voice presence only. |
| `guest` | Temporary status/chat only. |

Create and manage users:

```bash
theseus hive add-user --name "Alex" --role member --device-label "Alex iPhone"
theseus hive add-user --name "Parent" --role operator --device-label "Parent MacBook"
theseus hive users
theseus hive revoke-user alex
```

Generated user invites live under `reports/hive_user_invite_USER.json`. Treat
those files like passwords. User/operator invites are intentionally rejected by
the machine join installer path.

## Install And Run

Use the setup wizard for normal installs:

| Platform | Entry point |
| --- | --- |
| Windows | `bin/project-theseus-setup.cmd` or the shortcut created by `scripts/install_theseus_hive.ps1`. |
| macOS | `bin/project-theseus-setup.command`, or install once with `scripts/install_theseus_hive_macos.sh` to create the menu bar app and setup/doctor apps. |
| Linux | `scripts/install_theseus_hive_linux.sh`, then the generated desktop launcher or user service. |
| iPhone / Apple Watch / Android | `/mobile` PWA, setup wizard QR/deep link, or the native shell projects under `ios/TheseusHive` and `android/TheseusHive`. |

CLI wrapper:

```bash
./scripts/install_theseus_cli.sh
./bin/theseus.sh status
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_theseus_cli.ps1
.\bin\theseus.cmd status
```

Mac install/join example:

```bash
chmod +x scripts/install_theseus_hive_macos.sh scripts/start_theseus_hive.sh
./scripts/install_theseus_hive_macos.sh \
  --invite reports/hive_invite_private.json \
  --coordinator-url http://COORDINATOR-IP:8791 \
  --auto-update-soft \
  --install-service \
  --enable-service
```

Probe:

```bash
python3 scripts/hive_node.py probe \
  --out reports/hive_status.json \
  --peers-out reports/hive_peers.json
```

One-step join/bootstrap bundle:

```bash
theseus bootstrap \
  --out reports/hive_join_bundle.json \
  --qr-out reports/hive_join_profile_qr.svg

theseus join --invite reports/hive_join_bundle.json --start
```

The bootstrap bundle is the no-Codex onboarding object for spare Macs, Windows
PCs, Linux/RPi nodes, iPhones, and Watches. It contains LAN/tunnel/relay
endpoint candidates, Hive id/name/tier, token scope, update catalog URLs,
installer artifact URLs, install commands, and an iPhone deep-link profile. The
token-bearing bundle and QR code are passwords; generate `--no-token` copies
for screenshots or docs.

Mac canary gate before real training or installing spare Macs:

```bash
theseus mac roles --write-local-config
theseus mac training-preflight --execute --offline --allow-battery-smoke
theseus mac dmg-readiness --execute
theseus mac join-bundle
theseus mac app-status --text
theseus mac canary --execute --write-join-bundle
```

Mac gate contract:

- Apple Silicon Macs receive `mlx_training`, `mlx_eval`, `mlx_rollout`,
  operator, storage, artifact sync, and update-client roles only when MLX is
  actually visible to the runtime probe.
- Intel Macs receive `cpu_worker`, storage, operator, lightweight
  orchestration, artifact sync, and update-client roles; they must not
  advertise `mlx_apple`/`apple_mlx`.
- `training-preflight` rejects long training when runtime, thermal, disk, or
  battery gates are bad. With `--execute`, Apple Silicon queues one local
  `mlx_training_chunk`, waits for the worker report, and verifies no artifact
  sync, teacher use, or external inference occurred. `--allow-battery-smoke`
  only lowers the battery gate for this bounded canary; it does not allow long
  training on battery.
- `python3 scripts/autonomy_launch_readiness.py --profile smoke
  --require-teacher-cli` reports `ready_for_local_macos_smoke_training`
  separately from `ready_for_autonomous_training`. Local Mac smoke can be ready
  while full autonomy, candidate promotion, and legacy-port runtime promotion
  remain blocked. The operator API exposes this under `training.mac_local`.
- `scripts/coherence_delirium_metric.py` refreshes the local coherence source
  report consumed by `scripts/coherence_delirium_gate.py`; it uses local
  governance artifacts only and records `external_inference_calls=0`.
- `dmg-readiness --execute` rebuilds the DMG/pkg/zip/app, verifies/publishes
  the Hive catalog for the current commit, refreshes installer artifacts, and
  checks that the installed app serves update catalog and installer artifacts
  without Codex.
- `join-bundle` writes `dist/macos/ProjectTheseusHive.join.json`, a QR profile,
  and a double-click `Join Project Theseus Hive.command`. Token-bearing join
  bundles are passwords.
- `theseus train teacher-preflight --require-teacher-cli` verifies that full
  training can queue proposal-only teacher work without letting worker chunks
  use external inference. Add `--allow-teacher-live --require-live-teacher` to
  prove the real Codex teacher call path. The operator API exposes the result
  under `training.teacher`.

If `THESEUS_MACOS_JOIN_BUNDLE=/path/to/ProjectTheseusHive.join.json` is set
when building the DMG, the packager embeds that join profile in the app
payload. The macOS installer auto-applies a bundled join profile and then prints
post-install status lines: `Joined`, `Running`, `Update OK`, `Installer
Artifacts`, and `Training Ready`.

## Discovery

Nodes discover each other through three layers:

| Layer | Purpose |
| --- | --- |
| Signed multicast | Fast same-LAN/hotspot discovery with HMAC signatures when a join token/shared secret exists. |
| Bonjour/DNS-SD | macOS-native `_theseus-hive._tcp` advertisement and scan via `dns-sd` when available. |
| Coordinator/relay heartbeat | Durable cross-subnet, workshop, WireGuard, and travel-Hive sync. |

Peer state is written to `reports/hive_peer_registry.json` and summarized in
`reports/hive_peers.json` as `reachable`, `discovered`, `stale`, `blocked`, or
`unverified`. Stale records are retained for diagnostics, then aged out by the
configured retention window. Capability refresh includes CPU, memory, disk,
power/battery, thermal state, CUDA, MLX, storage, remote-control, voice,
roles, update readiness, and task slots.

Macs also export `.local` Bonjour candidates in the operator roaming profile,
ahead of private IP addresses. Native iPhone/Watch clients should try the last
good endpoint first, then Bonjour/local LAN, then private tunnel, then HTTPS
relay. Reverse-DNS `.arpa` names are filtered out because they are not useful
handoff targets.

Useful commands:

```bash
python3 scripts/hive_node.py discover --seconds 8 --out reports/hive_peers.json
theseus device list
theseus hive network-doctor --coordinator-url http://COORDINATOR-IP:8791
```

## Operator Surfaces

| Surface | Purpose |
| --- | --- |
| `/mobile` | Phone-friendly PWA for status, chat, tasks, storage, voice, training, and remote-control handoffs; it supports home-screen icons and token import through URL fragments so native wrappers do not leak tokens in request URLs. |
| `ios/TheseusHive` | Native SwiftUI wrapper around `/mobile`; iOS 16+ including iPhone X-class devices, Keychain token storage, native QR/deep-link/profile import, roaming endpoint failover, native peer/MLX/always-active status, and foreground polling of the Hive notification feed. Build readiness is checked with `scripts/build_theseus_ios.sh --validate`; signed TestFlight archives use the same script with `--archive`. |
| `ios/TheseusHive/TheseusHiveWatch` | Native watchOS 9+ companion embedded in the iPhone project; receives the private operator profile and notification relay through WatchConnectivity, stores its token in watchOS Keychain, shows Hive health/version/peer/work/MLX state, supports text/dictation chat, checkpoint and overnight prompts, local notifications, and haptics. |
| `android/TheseusHive` | Native Android wrapper around the same operator contract. |
| visionOS / Quest | Native clients should use the shared spatial contract first: profile import, room/node scene, voice route, storage anchors, work state, and governed session handoff. Spatial devices are operator surfaces before training workers. |
| macOS menu bar app | Status glyph plus quick open for `/mobile`, dashboard, logs, always-active sweep/pause/resume/stop controls, training round queueing, MLX smoke tasks, and service restart. |
| Windows tray app | Local tray/operator entry point for Windows nodes. |
| `theseus` CLI | Server/SSH/admin path for setup, users, profiles, storage, training, utilization, remote access, and updates. |

Core authenticated endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/hive/auth/status` | Cheap authenticated token/status probe for network doctor, schedulers, and clients that need to verify trust without building the full operator UI summary. |
| `GET /api/hive/operator/status` | Fleet, peers, roles, storage, accelerators, training, updates, and task summary. |
| `GET /api/hive/operator/roaming-profile` | Authenticated LAN/tunnel/relay endpoint profile for iPhone/Watch import and failover. It includes endpoint priority, Bonjour metadata, handoff policy, update catalog URLs, and installer artifact URLs. With `include_token=1`, it echoes only the token supplied by the current request. |
| `GET /api/hive/operator/notifications` | Authenticated, deduplicated operator notification feed from network, training, update, utilization, and task health. |
| `POST /api/hive/operator/notifications/ack` | Per-user notification acknowledgement. |
| `GET /api/hive/network-doctor` | Authenticated live network doctor for local API, coordinator, peers, stale records, and roaming posture. |
| `POST /api/hive/operator/chat` | Local/auto targets run the canonical assistant path with checkpoint memory, VCM context, deterministic tool/code evidence, and dogfood feedback; explicit remote targets still queue bounded `checkpoint_chat`. |
| `POST /api/hive/operator/assistant-feedback` | Record metadata-only post-response dogfood feedback (`accepted`, `missed`, `ignored`, `corrected`, or `completed`) for the latest operator assistant artifact and refresh local training rows. |
| `POST /api/hive/operator/task` | Queue an allowed registered task kind. |
| `GET /api/hive/spatial/status` | Privacy-preserving spatial scene for native visionOS/Quest/iPhone operator clients: rooms, zones, node placement, voice route, storage anchors, remote-control readiness, and training/work state. |
| `GET /api/hive/storage/*` | Browse/read configured storage shares if user role allows it. |
| `POST /api/hive/remote-control/session` | Create an audited remote-control handoff if user role allows it. |
| `GET /api/hive/voice/route` | Show current room listen/respond route. |
| `POST /api/hive/voice/presence` | Record score-only local presence; raw audio and transcripts are rejected. |
| `GET /api/hive/installer-artifacts` | Authenticated package/artifact index. |

## macOS Packaging

Build on a Mac:

```bash
THESEUS_MACOS_BUILD_DMG=1 ./scripts/package_theseus_macos.sh
```

Run the release-candidate gate before installing on spare Macs:

```bash
python3 scripts/macos_mlx_parity_audit.py \
  --out reports/macos_mlx_parity_audit.json \
  --markdown-out reports/macos_mlx_parity_audit.md
python3 scripts/macos_mlx_work_proof.py \
  --out reports/macos_mlx_work_proof.json \
  --markdown-out reports/macos_mlx_work_proof.md
python3 scripts/hive_macos_release_gate.py --execute
# or through the installed CLI:
theseus hive macos-release-gate --execute
```

The gate rebuilds the app/zip/pkg/dmg, verifies the current Hive version,
publishes the private update catalog, mirrors that catalog into the runtime
reports served by the LaunchAgent-backed node, reinstalls this Mac as the
Apple-Silicon or Intel canary, checks `/mobile`, `/api/hive/update-catalog`,
`/api/hive/installer-artifacts`, LaunchAgents, menu bar status, MLX capability,
and local soft-update convergence. It writes:

| Report | Path |
| --- | --- |
| JSON gate | `reports/hive_macos_release_gate.json` |
| Markdown gate | `reports/hive_macos_release_gate.md` |
| MLX parity audit | `reports/macos_mlx_parity_audit.json` |
| MLX parity audit notes | `reports/macos_mlx_parity_audit.md` |
| MLX runnable work proof | `reports/macos_mlx_work_proof.json` |
| MLX runnable work proof notes | `reports/macos_mlx_work_proof.md` |
| Shared node reachability registry | `reports/hive_node_registry.json` |
| Training fleet readiness | `reports/hive_training_orchestrator.json` |

The Mac runtime doctor is the first runtime gate. It identifies the active
shell Python, source venv, installed app venv, LaunchAgent programs/env, runtime
roots, mirrored verified version, update catalog, version status, update
check-in, local license-registration file presence, and join-config file
presence. It does not read or print local invite/user tokens. On Apple Silicon,
an active shell Python without MLX is reported as a false-negative context when
the source or installed Hive venv can import and execute `mlx.core`; that should
not block Mac gates by itself.

The MLX parity audit now separates static command coverage from live runnable
evidence. It records the latest registered MLX worker reports, MLX CLI bridge
reports, key metrics, receipts, promotion status, and any still-pending
Rust/Metal or Rust/MLX kernel ports. A `YELLOW` parity audit with
`runnable_evidence_missing=0` means the Python MLX bridges are real and tested,
but deeper native kernel parity is still tracked separately.
The report also writes `routing_decisions`: registered MLX worker chunks route
to Apple Silicon Macs, runnable MLX command bridges are allowed for bounded Mac
work, and CUDA-equivalent hot-loop parity proofs continue routing to
Windows/NVIDIA until the matching Rust/Metal or Rust/MLX port is marked ready.

The node registry is the scheduler-facing truth for remote placement. A remote
node can be trusted and still blocked for work if the local node does not have a
fresh outbound status path to it. Training placement requires outbound-verified
and non-flapping reachability; stale, blocked, or inbound-only peers remain
visible but are not eligible for queued CUDA/MLX chunks.

The network doctor verifies authentication through `GET /api/hive/auth/status`
first, then falls back to the richer `GET /api/hive/operator/status` only for
older peers. Slow operator summaries must not be interpreted as failed trust.
The macOS release gate also checks the same auth-status endpoint locally, and
the training link doctor uses it to prove worker trust before relying on the
heavier operator surface.

Outputs:

| Artifact | Path |
| --- | --- |
| App | `dist/macos/ProjectTheseusHive.app` |
| ZIP | `dist/macos/ProjectTheseusHive.zip` |
| PKG | `dist/macos/ProjectTheseusHive.pkg` |
| DMG | `dist/macos/ProjectTheseusHive.dmg` |
| Manifest | `dist/macos/hive-installer-artifacts.json` |
| Compatibility notes | `dist/macos/README-MACOS-COMPATIBILITY.txt` |
| MLX notes | `dist/macos/README-MLX.txt` |

Current compatibility contract:

- app minimum macOS: `10.15`;
- target architectures: `x86_64` and `arm64`;
- shell app wrapper is architecture-neutral;
- opening `ProjectTheseusHive.app` from the DMG installs the payload, creates
  the user LaunchAgents, enables safe soft auto-updates, and starts the Hive
  through launchd without a foreground start that can hold the installer window
  open;
- packaged/copied installs include `configs/hive_build_version.json` so nodes
  without `.git` or Codex still report the verified Hive commit/version for
  catalog convergence;
- source-dev Mac checks and installed LaunchAgent checks share the same default
  runtime root: `~/Library/Application Support/Project Theseus Hive/runtime`;
- Swift menu bar helper is built universal when possible;
- target installer validates helper architecture, recompiles locally with
  `swiftc` when needed, or installs a shell fallback;
- `configs/*.local.json` and `configs/*.secret.json` are excluded from packaged
  payloads so local join tokens and user hashes are not shipped.

Intel Macs are first-class CPU/storage/operator/relay nodes. They do not
advertise `mlx_apple`. Apple Silicon Macs can also advertise `mlx_apple` when
MLX is installed in the active Python environment.

Mac-native parity is tracked explicitly. `mlx_eval_chunk`,
`mlx_training_chunk`, and `mlx_rollout_chunk` are implemented worker chunks for
Apple Silicon. `symliquid-cli` also exposes `train-standalone-mlx`,
`train-rollout-mlx`, `train-rollout-mlx-sweep`, and
`train-token-superposition-mlx`; those commands run bounded first-party MLX
bridges through `scripts/macos_mlx_training.py`. The rollout worker is a
bounded MLX control probe, and the deeper Rust/Metal or Rust/MLX kernel ports
remain pending. Do not claim native Rust hot-loop parity from CPU fallback or
from the bridge alone. Check `reports/macos_mlx_parity_audit.json`
`routing_decisions` before deciding whether a lane should run on Apple
Silicon/MLX or Windows/NVIDIA CUDA.

Publish artifact metadata for Hive sync:

```bash
python3 scripts/hive_version_manager.py installer-artifacts \
  --out reports/hive_installer_artifacts.json
```

For broad private rollout, pass one Apple-Silicon canary and one Intel canary
with the gate above. For distribution outside your own machines, add Developer
ID signing and notarization; ad-hoc signing is only a private testing path.

## Remote Access

Preferred path:

1. Use LAN/hotspot when local.
2. Use a self-hosted WireGuard/private tunnel between home, workshop, and travel
   devices.
3. Use an authenticated Hive relay behind HTTPS when direct tunnel access is not
   practical.

Configure relay/tunnel URL:

```bash
theseus remote status
theseus remote wireguard-guide
theseus remote configure-relay --relay-url http://10.87.0.1:8793 --start
theseus remote mobile-profile
theseus remote doctor
```

Do not expose `8791`, `8787`, RDP, or VNC directly to the public internet. If a
relay must be reachable from the public internet, put HTTPS and firewall rules
in front of it and keep the Hive token/user token requirement enabled.

Before trusting multi-node work, run:

```bash
theseus hive network-doctor
```

The report is written to `reports/hive_network_doctor.json` and
`reports/hive_network_doctor.md`. A RED state means the Hive should not pretend
distributed work is real yet; fix live coordinator/peer reachability first.
The mobile operator has the same doctor button under Network. The doctor also
tracks directionality: a peer can be recently seen by inbound heartbeat while
still being blocked for outbound task queueing from this node. Endpoint probes
are retried with bounded latency telemetry; a coordinator or remote peer that
only succeeds after retry recovery, or that the peer registry marks as
`flapping`, is a RED distributed-work finding until repeated probes are stable.

The scheduler-facing registry embeds the latest doctor state under
`network_doctor` and mirrors compact readiness fields into `summary`, including
`network_doctor_state`, `network_doctor_red_finding_codes`,
`distributed_training_ready`, `mixed_cuda_mlx_training_ready`, and
`remote_cuda_live_ready`. The training orchestrator writes the same doctor
summary plus `fleet_readiness`, so operator/mobile clients can show the exact
reason CUDA or remote work was not queued. Authenticated status success is
treated as proof that the local Hive secret worked; older peers that omit
`security.shared_secret_configured` are labeled as reporting ambiguity, not
failed auth.

### Offline Flight Mode

When a Mac is away from the Hive LAN, use offline utilization mode instead of
pretending Windows/CUDA or remote artifact sync is available:

```bash
theseus solo loop --execute --allow-battery \
  --keep-awake \
  --min-battery-percent 35 \
  --profile inner_loop \
  --max-new-jobs 1 \
  --sleep-seconds 120 \
  --out reports/hive_solo_learning_status.json
```

`theseus solo` wraps the utilization manager in offline/local-only mode, then
writes the solo ledger, local best-by-arm activation manifests, stale-lease
recovery state, and overnight report inputs. The underlying worker chunks still
forbid teacher/external inference and record `external_inference_calls=0`.
`--allow-battery` must be explicit; the battery floor still blocks work once
the Mac drops below `--min-battery-percent`.
`--keep-awake` holds a macOS `caffeinate` assertion while the utilization
process runs, which prevents normal idle sleep during a long local run.

Closed-lid limit: a truly sleeping Mac cannot train. On MacBooks, reliable
closed-lid training requires normal macOS clamshell conditions, typically AC
power with an external display/input path. Without that, leave the lid open and
use `--keep-awake` for long runs.
Stop it with:

```bash
touch reports/hive_utilization_stop.flag
```

Solo/offline audit files:

| Artifact | Purpose |
| --- | --- |
| `reports/hive_solo_learning_status.json` | Current standalone state, MLX readiness, pause/stop controls, worker events, and next actions. |
| `reports/hive_solo_learning_ledger.jsonl` | Append-only local rows with input artifact, arm id, backend, score, output artifact, failure reason, and promotion decision. |
| `reports/hive_solo_best_by_arm.json` | Best local artifact per arm plus promotion history and rollback metadata. |
| `reports/hive_solo_overnight_report.json` / `.md` | What ran, improved, failed, was promoted, and should run next while offline. |
| `checkpoints/hive_promoted/<arm>/active_manifest.json` | The active local artifact pointer used as the next round's input artifact. |

## Node Discovery

Node discovery uses three paths together:

| Path | Purpose |
| --- | --- |
| Signed LAN multicast | Same-subnet nodes find each other quickly. If a Hive join token exists, multicast announcements are HMAC-signed and unsigned peers stay untrusted. |
| Authenticated heartbeat | Joined nodes POST `/api/hive/heartbeat` with the Hive secret, register themselves, and receive the node's current trusted peers. Inbound heartbeat proves the peer is alive, not necessarily callable from this node. |
| Outbound probe | Known peers are called back through `/api/hive/heartbeat`; only a successful callback is marked `reachable` for work routing. |
| Durable registry | Known peers persist in `reports/hive_peer_registry.json` and are re-probed after service restart, sleep, or network changes. |

`reports/hive_peers.json` uses clear states: `reachable` means this node has
successfully called the peer, `discovered` means recently seen but not yet
callable, `blocked` means a callback failed after discovery, and `stale` means
the peer aged out. Schedulers still perform a live API check before sending
work.

New node flow:

```bash
./scripts/install_theseus_hive_macos.sh \
  --invite reports/hive_invite_private.json \
  --coordinator-url http://COORDINATOR-IP:8791 \
  --auto-update-soft \
  --install-service \
  --enable-service \
  --start

theseus hive network-doctor
```

After install, the node should announce on LAN, register with the coordinator,
learn the current trusted peers, persist them locally, and start accepting only
registered bounded Hive tasks. If the coordinator cannot be reached, the
Network Doctor reports exact firewall, IP, VLAN, or stale-peer fixes.

## Storage And NAS Extensions

Hive storage is the non-screen-sharing path for file access. It exposes only
named shares that you explicitly configure.

```bash
theseus storage status
theseus storage add-share --path /Volumes/NAS/Photos --name Photos --tag photos
theseus storage browse --share-id photos
theseus storage pull --peer-url http://NODE:8791 --share-id photos --path IMG_0001.jpg
```

Rules:

- shares are read-only by default;
- arbitrary filesystem browsing is not exposed;
- credential folders, whole disks, and private config directories should not be
  shared;
- user roles can block storage browsing/downloads;
- phone `/mobile` can preview configured files without taking over the desktop.

Local config: `configs/hive_storage.local.json`.

## Remote Control

Remote control is a governed handoff, not raw Hive screen streaming.

```bash
theseus control status
theseus control request --target-node NODE --provider auto
theseus control launch --provider rdp --host NODE --execute
```

Supported provider families:

- RustDesk, preferably with private/self-hosted relay;
- Microsoft Remote Desktop/RDP for Windows over LAN/private tunnel;
- macOS Screen Sharing/VNC over LAN/private tunnel;
- Sunshine/Moonlight for low-latency paired hosts.

User roles can block remote-control sessions. The built-in Hive relay does not
proxy raw desktop streams.

## Voice Following

Voice following routes presence and response location across room nodes. It is
designed for “talk to the nearest Hive device” behavior without relaying raw
audio through the Hive API.

```bash
theseus voice configure-room --room-id kitchen --room-name Kitchen --microphone --speaker
theseus voice status
theseus voice presence --score 0.8 --source smoke
theseus voice route
```

Policy:

- score-only presence events;
- no raw audio relay by default;
- no transcript required for routing;
- external STT/TTS providers are forbidden for routing;
- future native STT/TTS can feed the same presence route after local policy
  gates are ready.

Local config: `configs/hive_voice_following.local.json`.

## Spatial Operator Contract

The spatial layer gives visionOS, Quest 3/OpenXR, iPhone, and desktop operator
clients the same Hive map without creating a new security boundary.

```bash
theseus spatial configure-node --room-id kitchen --room-name Kitchen --zone house-main --x 0 --y 0 --z 0 --yaw 0
theseus spatial status
```

The registered low-risk task kind `spatial_status` refreshes the same report for
always-active maintenance cycles.

`GET /api/hive/spatial/status` returns:

- rooms/zones and optional manually configured node coordinates;
- mic/speaker/display/operator-surface capability summaries;
- active voice-following route and confidence;
- nearby storage share summaries and explicit device tags;
- remote-control handoff readiness, not raw desktop streaming;
- training/work state anchors for seeing what the cluster is doing in space.

Privacy policy:

- no passthrough camera frames, raw room meshes, or raw mic audio are accepted
  or retained by the Hive API by default;
- clients exchange summaries such as `room=kitchen`,
  `speaker=node-mac-mini`, and `confidence=0.81`;
- file access still goes through explicit storage shares and role checks;
- spatial devices are high-bandwidth operator surfaces first, not training
  workers.

Local config: `configs/hive_spatial.local.json`.

## Training And Utilization

Hive training is decentralized and bounded. Any trusted node can plan a round;
workers only receive registered task kinds and clamped profiles.

Worker task kinds:

| Task kind | Preferred backend |
| --- | --- |
| `cuda_eval_chunk` | Windows/Linux NVIDIA |
| `cuda_training_chunk` | Windows/Linux NVIDIA |
| `cuda_rollout_chunk` | Windows/Linux NVIDIA Rust/CUDA rollout path |
| `mlx_eval_chunk` | Apple Silicon Mac with MLX |
| `mlx_training_chunk` | Apple Silicon Mac with MLX |
| `mlx_rollout_chunk` | Apple Silicon Mac with MLX rollout/control probe |
| `training_smoke` | CPU/CUDA/MLX small validation |
| `training_orchestrate` | Local planner/scheduler action |

Mac MLX command bridge:

| Command | Used for |
| --- | --- |
| `train-standalone-mlx` | Apple MLX readout training bridge. |
| `train-rollout-mlx` | Apple MLX rollout/control worker bridge. |
| `train-rollout-mlx-sweep` | Apple MLX rollout sweep bridge. |
| `train-token-superposition-mlx` | Apple MLX token-superposition readout bridge. |

Commands:

```bash
theseus hive training-link --refresh
theseus train status
theseus train plan
theseus train run --sync-artifacts
theseus train sync
theseus train overnight
theseus solo status
theseus solo sweep --execute --allow-battery --max-new-jobs 1
theseus solo loop --execute --allow-battery --keep-awake --sleep-seconds 60 --max-new-jobs 1
theseus solo overnight
theseus utilize status
theseus utilize sweep --execute
theseus utilize loop --execute --sleep-seconds 60 --max-new-jobs 2
theseus utilize loop --execute --keep-awake --sleep-seconds 60 --max-new-jobs 2
```

Always-active policy:

- user/operator work has first claim on queue and running slots;
- idle CUDA/MLX slots get bounded decentralized training rounds by arm;
- idle CPU slots rotate bounded smoke training, readiness, storage indexing,
  capability, network doctor, voice, update, and version-status tasks;
- battery-powered nodes are deprioritized/blocked for background utilization
  unless local policy explicitly allows battery use;
- on macOS, `--keep-awake` prevents idle sleep for the lifetime of the
  utilization process but does not make a sleeping or unsupported closed-lid
  Mac continue computing;
- if no safe worker slot is available, a grounded checkpoint summary keeps the
  operator view fresh;
- stop/pause flags, resource floors, licensing, and public-data guards override
  the no-downtime goal.

Use `theseus utilize loop --execute` for a foreground overnight queue-filler.
Use `theseus solo loop --execute` when the Mac is away from the Hive LAN or you
want auditable local-only MLX/CPU improvement without artifact sync.
Installed vacation-mode services run the same utilization sweep once per cycle.
The mobile operator shows the same always-active state, per-node coverage,
recent sweep summary, solo-learning state, local promotions, and
pause/resume/stop/sweep controls.

Overnight audit:

- `theseus train overnight` writes `reports/hive_overnight_training_report.json`
  and `.md`;
- `theseus solo overnight` writes `reports/hive_solo_overnight_report.json`
  and `.md` for standalone/offline Mac runs;
- each worker row includes input artifacts, output artifacts, score, arm id,
  owner node, backend, merge result, and promoted model path when available;
- expired leases are recoverable: the next deterministic round replans against
  currently reachable nodes so sleeping/offline nodes do not strand an arm.

Training guardrails:

- no public benchmark solutions as training data;
- no teacher apply mode inside worker chunks;
- no arbitrary shell;
- accepted artifacts are synced and merged through report/ledger paths;
- local worker reports and peer-fetched worker reports are both eligible for
  best-by-arm promotion through `reports/hive_artifact_merge_summary.json`;
- long tasks should be selected by resource slots, latency, and backend fit.

Core reports:

- `reports/hive_training_orchestrator.json`
- `reports/hive_training_state.json`
- `reports/hive_training_orchestrator_ledger.jsonl`
- `reports/hive_training_link_doctor.json`
- `reports/hive_overnight_training_report.json`
- `reports/hive_network_doctor.json`
- `reports/hive_utilization_status.json`
- `reports/hive_utilization_plan.json`

## Rented Compute And Storage

Rented capacity is plan-first and explicit-execute only.

```bash
theseus rent status
theseus rent init --provider aws --name aws-gpu-nightly
theseus rent plan --profile aws-gpu-nightly --task-kind cuda_training_chunk --hours 4
theseus rent launch --plan reports/hive_rented_compute_plan.json --execute
theseus rent stop --profile aws-gpu-nightly --instance-id i-... --execute
```

Provider scaffolds include AWS, GCP, Azure, RunPod, Vast, and object-storage
profiles for AWS S3, GCP GCS, and Azure Blob. Plans check budget, time window,
queue pressure, local disk floor, transfer pressure, credential posture, and
public-data leakage constraints before launch.

The rented-compute code is split by responsibility: `scripts/hive_rented_compute.py`
owns status/plan/launch/stop orchestration, while
`scripts/hive_rented_compute_profiles.py` owns provider aliases, defaults,
profile normalization, catalog rows, and provider CLI readiness. Keep new
providers in the profile module first, then promote launch behavior in the
orchestrator only after dry-run evidence exists.

Local config: `configs/hive_rented_compute.local.json`.

## Updates And Release Flow

Soft updates synchronize metadata and verified catalogs. Hard source/app
replacement remains package-based or explicit.

```bash
theseus update verify-hive
theseus update publish-hive
theseus update converge-hive --execute
```

Workers installed with `--auto-update-soft` configure a local periodic updater:

- Windows Scheduled Task;
- macOS LaunchAgent;
- Linux user systemd timer when available.

Installer artifacts are exposed through:

```text
GET /api/hive/installer-artifacts
```

## Important Files

| Path | Purpose |
| --- | --- |
| `configs/hive_policy.json` | Main policy: ports, roles, tasks, multi-user roles, routing, storage, voice, utilization, rented capacity. |
| `configs/hive_join.local.json` | Ignored active machine join config and token. |
| `configs/hive_users.local.json` | Ignored user/operator token hashes, roles, device labels, expirations, revocations. |
| `configs/hive_profiles.local.json` | Ignored saved Hive profiles. |
| `scripts/hive_node.py` | Node daemon, HTTP API, peer discovery, operator PWA, task worker. |
| `scripts/hive_users.py` | User/operator token creation, hashing, roles, revocation, invite output. |
| `scripts/hive_invite.py` | Machine invite creation/apply. |
| `scripts/macos_runtime_doctor.py` | Standalone Mac runtime doctor for source/app Python, MLX, LaunchAgents, runtime roots, and source-vs-installed version/catalog state. |
| `scripts/hive_storage.py` | Explicit storage shares and peer file proxy helpers. |
| `scripts/hive_remote_control.py` | Remote-control readiness and session handoff. |
| `scripts/hive_voice_following.py` | Room presence and route policy. |
| `scripts/hive_training_orchestrator.py` | Decentralized bounded training planner/runner. |
| `scripts/hive_utilization_manager.py` | Safe idle-slot filler. |
| `scripts/hive_rented_compute.py` | Rented compute/storage plan/launch/stop. |
| `scripts/hive_rented_compute_profiles.py` | Rented compute/storage provider aliases, defaults, catalogs, and profile normalization. |
| `scripts/package_theseus_macos.sh` | Universal-intent macOS app/pkg/dmg packager. |
| `reports/hive_status.json` | Local node status/capabilities. |
| `reports/hive_operator_status.json` | Last operator API report. |
| `reports/hive_peers.json` | Peer registry. |
| `reports/hive_installer_artifacts.json` | Cross-platform installer artifact index. |

## Security Rules That Should Not Regress

- Registered task kinds only.
- No arbitrary remote shell.
- No public dashboard/Hive API exposure without private tunnel or HTTPS relay
  and token auth.
- No ROM fetching/import as a remote Hive task.
- No bulk or uncertain-license data ingestion.
- No teacher oracle/apply inside worker chunks.
- No raw audio relay for voice following.
- No packaging of `configs/*.local.json` or `configs/*.secret.json`.
- No remote hard app/source replacement unless it is an explicit package/update
  flow.
