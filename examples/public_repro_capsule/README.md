# Public reproducibility capsule

This tiny, fully synthetic Apache-2.0 fixture exercises the Theseus evidence
protocol without publishing private curricula, heldouts, user data, or useful
model checkpoints. It performs deterministic training, writes model and
optimizer state, resumes from the midpoint, compares the resumed and
uninterrupted states exactly, generates an opaque candidate packet, verifies it
under a declared tolerance, and records file digests and measured runtime.

Run:

```bash
python3 scripts/public_reproducibility_capsule.py \
  --out-dir /tmp/theseus-public-repro \
  --gate
```

The result is protocol evidence only. It is not a model-capability,
architecture-selection, performance, or production-readiness claim.
