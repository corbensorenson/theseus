# SymLiquid FEP-Net

## A Compact Generative System for Active Compression, Symbolic Memory, and Free-Energy-Minimizing Agents

Version 3.0 implementation-aligned draft.

Status: retired conceptual substrate draft, moved to
`deprecated/docs/background/` on 2026-06-03. For the current implementation
state, autonomy stack, benchmark frontier, and training gates, read
`../../../docs/PROJECT_STATE.md` first. For the current top-to-bottom
whitepaper, use `../../../docs/PROJECT_THESEUS_WHITEPAPER.md`.

## Abstract

SymLiquid FEP-Net is a modular software architecture for agents that maintain streaming continuous state, bind compositional symbolic memory, expose inspectable nonlinear transformations, and select actions with an active-inference-inspired objective. This repository implements the smallest rigorous prototype of that idea in Rust: a CPU reference backend, toy tasks, ablation switches, CGS accounting, verification reports, tests, and an optional CUDA backend surface for later acceleration.

The project does not claim to replace transformers, deliver AGI, guarantee lifelong learning, or provide a fixed efficiency multiplier. Its value should be judged by measured behavior on controlled tasks, residual accounting, verification contracts, and ablations that isolate each architectural component.

## Compact Generative Systems Contract

A Compact Generative System is represented as:

```text
C = (S, R, M, epsilon, V, G)
```

where `S` is seed, `R` is rule system, `M` is memory, `epsilon` is residual, `V` is verification, and `G` is governance interface.

The implementation makes this concrete with:

| CGS field | Rust representation |
| --- | --- |
| Seed | task-specific compact state or belief |
| Rule | model parameters, transition rules, kernels |
| Memory | continuous, reservoir, VSA, or belief memory |
| Residual | error, failed retrieval, uncertainty, failure rate |
| Verification | `VerificationReport` |
| Governance | action/query/readout behavior |

The files are:

```text
../../../crates/symliquid-core/src/cgs.rs
../../../crates/symliquid-core/src/eval.rs
../../../docs/CGS.md
```

## Core Claim

The architecture couples four state spaces:

| State space | Role |
| --- | --- |
| Continuous liquid state | adaptive temporal context |
| Reservoir state | cheap nonlinear memory expansion |
| Vector-symbolic memory | role-filler binding, bundling, unbinding, cleanup |
| Belief state | uncertainty over hidden causes and action outcomes |

KAN-lite layers provide inspectable scalar edge functions between these spaces. Expected-free-energy scoring provides a minimal policy-selection mechanism for active information seeking.

## Reference Implementation

The first implementation is Rust-first and CPU-runnable. PyTorch is not a runtime dependency. CUDA acceleration is intentionally scoped to specialized kernels rather than a full neural-network framework.

```text
Observation
  -> KAN-lite encoder
  -> liquid state update
  -> reservoir update
  -> VSA projection and memory update
  -> belief extraction / FEP scoring
  -> readout or action selection
```

## Synthetic Benchmarks

The included tasks are intentionally small:

| Task | What it tests |
| --- | --- |
| Role-filler binding | VSA binding, bundling, unbinding, cleanup retrieval |
| Delayed recall | recurrent state and reservoir memory under delayed query |
| Active classification | epistemic inspection before classification |
| Active gridworld | belief update, inspection, and action governance |

These tasks are not proof of broad intelligence. They are interface tests that make the architecture falsifiable.

## Ablations

Implemented ablation switches include:

- `no_kan`
- `no_liquid`
- `no_reservoir`
- `no_vsa`
- `no_fep`

The current CLI exposes task-relevant ablations:

```bash
cargo run -p symliquid-cli -- ablations --task role_filler
cargo run -p symliquid-cli -- ablations --task delayed_recall
cargo run -p symliquid-cli -- ablations --task active_classification
cargo run -p symliquid-cli -- ablations --task gridworld
```

## CUDA Direction

The CUDA crate includes kernel sources for the primitives most likely to benefit from custom acceleration:

- VSA binding
- VSA bundling
- VSA permutation
- cleanup similarity search
- reservoir update
- KAN-lite RBF expansion
- liquid update
- expected-free-energy scoring

The current Rust CUDA surface launches the first real CUDA VSA kernels through `cudarc`/NVRTC while preserving CPU fallbacks and parity tests. Reservoir, liquid, KAN-lite, cleanup, and FEP CUDA launch paths remain the next acceleration targets; speedups should be reported only after benchmark comparison.

## Roadmap

1. Validate CPU tests and toy tasks on a machine with Rust installed.
2. Add real CUDA driver integration behind `--features cuda`.
3. Expand ablation coverage so every task can disable every major component.
4. Add serialization for configs and measured reports.
5. Add larger controlled benchmarks only after the toy tasks are stable.
