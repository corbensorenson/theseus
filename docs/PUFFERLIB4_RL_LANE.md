# PufferLib 4 RL Lane

Theseus should use PufferLib 4 as a fast RL pressure lane when the runtime is
ready, but the lane is governed by the same evidence and licensing rules as the
rest of the hive.

## Policy

- Use permissive/open local environments first: Puffer/Ocean, project-local
  synthetic RL, chess/Go self-play, and long-horizon tool-use traces.
- Do not download Atari or other commercial ROMs automatically.
- Atari/ALE is disabled until ALE is installed, explicit legal user-supplied or
  permissively licensed assets are present, and
  `configs/allow_user_supplied_atari.flag` exists.
- Store runtime/cache/training artifacts under `D:/ProjectTheseus`.
- Treat public benchmark surfaces as calibration-only.

## Runtime Gate

Run:

```powershell
python scripts\pufferlib4_capability_probe.py --out reports\pufferlib4_capability_probe.json --markdown-out reports\pufferlib4_capability_probe.md
```

The lane is GREEN only when:

- `pufferlib` imports.
- `pufferlib._C` native backend imports.
- At least one local Ocean environment with `binding.c` is available.

If the native backend is missing, the lane is YELLOW and emits a
`pufferlib_native_backend_missing` residual rather than wasting training time.

The first native backend build should target a tiny permissive Ocean
environment such as `cartpole`:

```bash
cd vendor/pufferlib
bash build.sh cartpole --cpu
```

On Windows, build through a real Linux/mac builder node or an approved Podman
Linux VM. Do not use Atari training until the explicit flag and legal assets
gate are satisfied.

## Training Lane

Run:

```powershell
python scripts\pufferlib4_rl_lane.py --probe --out reports\pufferlib4_rl_lane.json --markdown-out reports\pufferlib4_rl_lane.md
```

The lane emits metadata-only STS capsules for:

- legal action masks
- state memory
- sparse reward credit assignment
- branching plan selection
- reset/step contracts
- rollout replay
- policy/value traces
- repair after loss

Those capsules feed back into code decoder contracts, repo repair,
long-horizon tool use, and conversation-while-working behavior.
