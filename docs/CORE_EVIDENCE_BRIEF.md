# Project Theseus: First Flagship Evidence Result

## The answer

The governance mechanics replay cleanly; the current frozen worker cannot yet do the work needed to test stack efficacy.

This is a useful negative result. Theseus reproduced one exact local
allowed/block/revoke/rollback packet from a clean archive, but its frozen
deterministic repository worker produced plans rather than patches. It completed
**0 of 3** development tasks. The preregistered rule therefore stopped E2 before
opening its four held-out tasks. On a separate six-task repeated-work cohort,
all **42** candidate variants remained target-blind and none completed a task.

```mermaid
flowchart LR
  P["Natural request + parent snapshot"] --> W["Frozen local worker"]
  W --> G["Governance / route policy"]
  G --> V["Independent hidden evaluator"]
  V -->|useful + safe| E["Temporary effect + exact rollback"]
  V -->|plan-only / malformed| H["Hold + residual"]
```

## Matched results

| Experiment | Denominator | Useful | Result |
| --- | ---: | ---: | --- |
| Clean authority/effect replay | 1 packet | mechanics replayed | `POSITIVE_SCOPED` mechanics |
| E2 development competence | 3 tasks | 0 | `INCONCLUSIVE_WORKER_INADEQUATE` |
| E2 heldout | 4 tasks | unopened | preserved by frozen stop rule |
| E3 repeated work | 6 tasks / 42 variants | 0 | `INCONCLUSIVE_WORKER_INADEQUATE` |
| Existing-owner regressions | 9 controls | 9 passed | mechanics only |

![Useful-safe frontier](assets/core-evidence-useful-safe-frontier.svg)

The apparent safety of every E2/E3 arm is not a win: no candidate emitted an
effect-capable patch. Maximal routing consumed **114**
cost units, cheapest routing consumed **60**, and
least-sufficient routing held all six tasks because none met the frozen quality
predicate.

## What the evidence supports

- Exact local authority separation, observation, residual recording, and
  rollback mechanics are replayable for this packet.
- Candidate/evaluator separation held: E2 sealed 3 candidates and E3 sealed 42
  variants before opening their corresponding targets.
- The evidence process behaved correctly under failure: it retained zero-result
  denominators, did not lower the floor, did not open E2 heldout, and issued
  claim-scoped dispositions.

## What it does not support

It does not yet support governed-stack efficacy, planning utility, VCM utility,
procedural-reuse efficiency, least-sufficient routing, learned generation, or
student competence. D2 and public calibration remained untouched; there were
zero external-inference or teacher calls.

The 95% Wilson interval for the E2 useful rate is
**0.000–0.561**.
The denominator is deliberately shown because 0/3 is a wall, not a universal
falsification.

## Fastest meaningful next move

Do not add another governance surface and do not weaken the evaluator. Build or
qualify one genuinely patch-producing **local** worker on the already-open
development partition. It must emit a real unified diff that applies to the
parent snapshot and passes the independent hidden evaluator. Only after the
frozen competence floor is met should the unopened E2 heldout be consumed.

## Reproduce

```bash
python3 scripts/core_evidence_campaign.py --stage E2 --gate
python3 scripts/core_evidence_campaign.py --stage E3 --gate
python3 scripts/core_evidence_synthesis.py --gate
```

Evidence digests:

- E2: `d0dc3fde921435de00324bc8bb67ff9008a18805b736f6af94b3f266e07c5378`
- E3: `dd8a923f70e21c94a24b91d7429f9ed90b625e69db6767da950c9ad55c40cd77`
- E4: `7a59a90bb48ffc27bc99ec4eb8ffd76e739c28a9580a3e39b239d22dd3b88d87`

No ASI Stack book support state changes automatically from this brief.
