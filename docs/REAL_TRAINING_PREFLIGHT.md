# Real Training Preflight

This runbook defines the checks before a state-changing neural training action.
It does not authorize training. Current state is in
[Project State](PROJECT_STATE.md); execution order is in the
[roadmap](../roadmap.md).

## Current Posture

As of 2026-07-29:

- the exact shared trunk is at step 11,416 and 87,441,996 optimizer positions;
- model, AdamW, MLX RNG, cursor, and the 37-manifest prospective lineage are
  custody-green;
- the operator hold is installed;
- the model is `NOT_EVALUATED`;
- both matched dense controls are untrained;
- the selected route is compiled FP32 MLX;
- KERC/RDC and ANE are not campaign-one training paths;
- the current pre-training architecture gate is not ready after legitimate
  source changes.

Therefore no long run should launch from the current source state.

## Required State Transition

The only valid sequence is:

```text
documentation/source transaction committed
  -> source-bound architecture freeze replay
  -> independent readiness audit
  -> exact replacement package
  -> roadmap pre-training gate passes
  -> operator reviews evidence
  -> operator explicitly removes hold
  -> one bounded fresh-process segment
  -> exact transactional checkpoint
  -> review before another segment
```

A GREEN report cannot remove the hold.

## Invariants

Before every launch, verify:

1. **Exact candidate identity**
   - `moecot_mlx_57m_active_preregistered_v1`;
   - expected model/data/objective/schedule/evaluation identities;
   - no unreviewed topology or vocabulary change.
2. **Exact checkpoint custody**
   - model, optimizer, RNG, cursor, receipt, and lineage digests match;
   - the prospective anchor gap remains explicit;
   - no missing or rewritten append-only manifest.
3. **Training authority**
   - current architecture freeze passes;
   - current independent audit passes;
   - current replacement package passes;
   - roadmap gate passes with `--require-pre-training-ready`;
   - candidate-specific lease is valid.
4. **Operator authority**
   - the runtime hold was explicitly removed for this campaign action;
   - no inherited teacher, network, or remote-execution authority.
5. **Data**
   - frozen admitted corpus identity matches;
   - public benchmark payloads are absent;
   - provenance, licenses, deduplication, contamination, privacy, retention,
     tokenizer, and synthetic-share gates pass;
   - teacher share and optimizer-sampling caps pass.
6. **Evaluator**
   - functional freeze identity matches;
   - consumed case count is still zero before final campaign evaluation;
   - forbidden-field runtime enforcement and independent integrity checks pass.
7. **Resources**
   - disk reserve is derived from two complete measured checkpoint
     transactions;
   - host telemetry is available;
   - no arbitrary memory or disk floor is introduced;
   - causal process, swap, predicted-exhaustion, thermal, and write checks pass;
   - no competing accelerator job makes the observation uninterpretable.
8. **Execution**
   - fresh process;
   - external watchdog;
   - transactional segment boundary;
   - atomic checkpoint before yield;
   - no suspension of an in-flight Metal graph;
   - exact before/after child and host-guard receipts.
9. **Claims**
   - no capability, support, or utility claim from training progress;
   - any failure remains scoped to the exact run;
   - no nearby green artifact substitutes for a blocked owner.

## Readiness Commands

Run only after the documentation/source transaction is committed:

```bash
python3 scripts/pretraining_architecture_freeze.py --execute-replays
python3 scripts/pre_long_run_independent_readiness_audit.py
python3 scripts/pre_long_run_replacement_freeze.py
python3 scripts/roadmap_implementation_gate.py --gate --require-pre-training-ready
```

Then verify the repository and hold state:

```bash
git status --short
test -f runtime/control/neural_seed_yield_after_segment
```

At this point the hold should still exist. A passing readiness package means
`TRAINING_READY_BUT_HELD`.

## Bounded Qualification

When the operator can dedicate the laptop and explicitly authorizes the next
step, run one bounded campaign invocation rather than an unobserved open-ended
process:

```bash
python3 scripts/neural_seed_training_campaign.py --execute --max-segments 1
```

This command is valid only after the registered hold-removal procedure and all
current gates pass. Do not copy it into automation that bypasses those checks.

After the segment:

- verify optimizer positions rather than only nominal steps;
- rehash model, optimizer, RNG, cursor, and receipt;
- verify the appended lineage manifest;
- inspect host/swap/thermal/write evidence;
- confirm the functional surface remains unconsumed;
- restore or retain the hold before unrelated work.

## Full Campaign Order

After the bounded qualification:

1. complete the modular shared-trunk candidate;
2. train the dense-active-parameter control;
3. train the dense-total-parameter control;
4. verify matched raw data, compute, tuning, inference, verifier, and total
   system cost;
5. consume the frozen 160-case private functional evaluation once;
6. record model-only and assisted outcomes separately;
7. make the architecture decision from paired utility and cost.

Do not change the recipe based on intermediate numbers. A changed recipe is a
new candidate and lineage.

## Stop Conditions

Stop and preserve state on:

- digest, cursor, plan, lease, or lineage mismatch;
- nonfinite loss or gradients;
- checkpoint transaction failure;
- unavailable telemetry;
- causal resource-policy stop;
- unexpected evaluation consumption;
- public-data or forbidden-field fault;
- authority widening;
- integrity or verifier failure.

Elapsed hours, time of day, a round-number free-memory floor, or a desire for a
green report are not stop conditions.

## Non-Claims

Preflight proves only that a specific training transaction is permitted and
observable. It does not prove:

- that the model will be useful;
- that MoECOT will beat dense controls;
- that the selected route is fastest possible on every host;
- that KERC, ANE, or SymLiquid is invalid;
- that local security is adequate for LAN or internet exposure.
