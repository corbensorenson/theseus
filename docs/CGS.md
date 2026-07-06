# Compact Generative Systems

Compact Generative Systems, or CGS, is the design frame used by this repository.

The short definition:

```text
A compact, structured, verifiable core reconstructs, predicts, generates,
controls, or governs a larger system through rules, memory, residual correction,
and verification.
```

The implementation contract is:

```text
C = (S, R, M, epsilon, V, G)
```

| Term | Meaning | SymLiquid instance |
| --- | --- | --- |
| `S` | seed | observation, encoded feature, belief |
| `R` | rule system | KAN, liquid, reservoir, VSA, FEP transitions |
| `M` | memory | continuous state, reservoir state, VSA memory |
| `epsilon` | residual | prediction error, retrieval error, action failure |
| `V` | verification | tests, ablations, task checks, trace checks |
| `G` | governance | prediction, query, action, memory update |

## Why This Matters

SymLiquid is not just a list of modules. It is wired as:

```text
continuous state -> nonlinear expansion -> symbolic recompression
    -> belief/action governance -> residual correction
```

This is the CGS road:

```text
compress causal structure into verified governance handles
```

The code expresses that by requiring task reports to carry:

- residual cost
- verification status
- governance power
- seed/rule/memory costs
- generative leverage
- a provisional CGS quality score

The relevant code is:

```text
crates/symliquid-core/src/cgs.rs
crates/symliquid-core/src/eval.rs
crates/symliquid-core/src/tasks/
```

## Entry Criteria

A serious CGS should have:

1. Compact seed.
2. Explicit expansion or governance rules.
3. Persistent memory or state.
4. Residual accounting.
5. Verification contract.
6. Measured generative or governance output.
7. Full cost accounting.

If any of these are missing, the system is only a metaphor, not a CGS implementation.

## SymLiquid Thesis

SymLiquid FEP-Net instantiates CGS as an agent architecture:

```text
KAN feature compression
  -> liquid temporal compression
  -> reservoir nonlinear expansion
  -> VSA symbolic recompression
  -> belief update
  -> expected-free-energy governance
  -> residual verification and correction
```

The first prototype is intentionally small. Its job is to make the claim falsifiable, not to claim frontier model performance.
