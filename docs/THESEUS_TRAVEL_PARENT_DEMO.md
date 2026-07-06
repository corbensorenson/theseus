# Project Theseus Travel Demo Runbook

Audience: parents or curious non-specialists who use a little AI and want to
understand what this project is.

Target machine: Apple Silicon MacBook Pro with 16 GB unified memory.

## Demo Promise

Show Theseus as an honest self-improvement research system:

- It checks whether its own evidence is current.
- It refuses to promote itself when the gates are not satisfied.
- It names the remaining walls instead of pretending they are solved.
- It can show a private A/B improvement without using public benchmark answers.

Do not present it as ASI. Present it as a serious local learning scaffold that
is being engineered toward stronger self-improvement.

## Mac Setup

Run these from the project root on the MacBook:

```bash
python3 -m venv .venv-demo
source .venv-demo/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install mlx
python3 scripts/travel_demo_preflight.py --mode parents_demo --target apple_mlx
```

MLX is the preferred Apple Silicon path. CUDA remains the Windows/NVIDIA hot
path. On the 16 GB M1, use MLX for tiny local tensor/inference checks and use
cached Theseus reports for the demo. Do not start long training.

## Five-Minute Demo

1. Run:

```bash
python3 scripts/travel_demo_preflight.py --mode parents_demo --target apple_mlx
```

2. Open:

```text
reports/travel_demo_preflight.md
```

3. Say:

> This is Theseus checking itself before I show it to you. It knows whether the
> Mac backend is ready, whether its code-transfer evidence is fresh, and whether
> the dangerous switches like public calibration, model growth, and promotion
> are locked.

4. Open:

```text
reports/code_transfer_governance_remaining_walls.md
```

5. Say:

> The interesting part is not that it claims to be done. It says exactly what is
> not done. The current walls are adapter/runtime handling, interface fidelity,
> return-shape contracts, algorithm planning, and verifier quality.

6. Show the private A/B numbers from the preflight or remaining-wall report:

```text
private A/B pass delta: +0.375
private A/B no-admissible delta: -0.375
```

7. Say:

> That means a source-level private fix changed behavior in the intended
> direction, without training on public benchmark answers.

## Twenty-Minute Demo

Use the five-minute path, then add:

- `reports/public_calibration_readiness_packet.md`
- `reports/decoder_v2_private_ablation_gate.md`
- `reports/private_public_transfer_proof.md`
- `reports/maturity_integrity_audit.md`

Talk track:

- Readiness packet: canonical broad-floor v2 evidence and lock state.
- Decoder gate: candidate coverage, STS-conditioned candidates, no-admissible
  rate.
- Transfer proof: private-to-public receiver coverage improved, still without
  executing public benchmark training.
- Maturity audit: the system stays YELLOW because broad public transfer is not
  above the floor yet.

## What Not To Run Live

Avoid these during the parent demo:

- Public calibration.
- Long Code LM training.
- Model growth.
- Candidate promotion.
- GPU/MLX stress tests.
- Large local LLMs on the 16 GB M1 while screen-sharing.

The MacBook demo should feel fast and calm. It should not make the fans, memory
pressure, or terminal errors become the star of the room.

## If MLX Is Not Ready

If the preflight says `mlx_not_importable`, run:

```bash
python3 -m pip install mlx
python3 scripts/travel_demo_preflight.py --mode parents_demo --target apple_mlx
```

If MLX still fails, use the offline fallback:

- Open `reports/travel_demo_preflight.md`.
- Open `reports/code_transfer_governance_remaining_walls.md`.
- Explain that the Mac backend setup is separate from the governance evidence.

## Good One-Sentence Explanation

> Theseus is my local research system for making AI improve itself honestly:
> it runs experiments, checks its own evidence, blocks itself from cheating, and
> turns failures into the next private architecture fixes.

