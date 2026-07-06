---
name: project-theseus-core-values
description: Corben's hard constraints and end goal for Project Theseus — no external inference served to users, teacher-in-training only, personal seed models that self-improve
metadata:
  type: project
---

Stated by Corben on 2026-06-12: it is a **betrayal** of Project Theseus to use external inference anywhere except as a teacher during training. External/large-model output must never be served to the end user in any way.

**End goal:** personal models that only know what they need to best help the human using them. Train an initial seed to the point where it can self-improve well enough on its own; paid external (teacher) inference is acceptable only short-term during that bootstrap.

**Why:** privacy, ownership, and cost — the user does not want long-term dependence on paid external inference.

**How to apply:** never propose serving open-weights or API models to end users, even wrapped in the VIEA harness. Routing real work to external/large-model "arms" (execution-plane proposals) was explicitly rejected. Capability gains must come through: licensed open corpora, teacher distillation during training, and self-generated verifier-checked data. The repo's `external_inference_calls=0` serving gates reflect this value and must stay.
