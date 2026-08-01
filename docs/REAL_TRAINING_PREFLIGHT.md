# Real Training Preflight

This runbook defines the checks before a state-changing neural training action.
It does not authorize training. Current state is in
[Project State](PROJECT_STATE.md); execution order is in the
[roadmap](../roadmap.md).

## Current Posture

As of 2026-07-30:

- the exact shared trunk is at step 11,416 and 87,441,996 optimizer positions;
- model, AdamW, MLX RNG, cursor, and the 37-manifest prospective lineage are
  custody-green;
- the optional emergency-yield control is absent;
- the model is `NOT_EVALUATED`;
- both matched dense controls are untrained;
- the selected route is compiled FP32 MLX;
- KERC/RDC and ANE are not campaign-one training paths;
- the current independent readiness audit is GREEN after recomputing exact and
  normalized prompt freshness against the consumed v8 packet;
- the replacement freeze is RED only on its mandatory clean-source binding;
- no 57M D2 result exists; the machine-predicate one-shot D2 controller is
  implemented, but the surface remains sealed until its clean-source,
  exact-freeze, checkpoint-completion, local-rater, nonconsumption,
  output-collision, and exclusive-accelerator predicates all pass.

Therefore no long run should launch from the current source state.

## Required State Transition

The only valid sequence is:

```text
documentation/source transaction committed
  -> source-bound architecture freeze replay
  -> independently prove the D2 surface is fresh and private
  -> independent readiness audit
  -> exact replacement package
  -> roadmap pre-training gate passes
  -> autonomous controller reacquires every machine predicate
  -> controller acquires an exclusive one-shot lease without touching the
     emergency-yield control
  -> exact transactional checkpoint
  -> controller releases the lease and reevaluates before another segment
```

A GREEN report cannot bypass an emergency-yield request. Only the registered
autonomous controller can acquire the exclusive execution lease for one
transaction, and no user or operator approval is part of that decision.

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
4. **Machine authority**
   - the autonomous controller acquired an exclusive one-shot lease after
     recomputing every launch predicate;
   - the optional emergency-yield control is never removed or modified;
   - no inherited teacher, network, or remote-execution authority.
5. **Data**
   - frozen admitted corpus identity matches;
   - public benchmark payloads are absent;
   - provenance, licenses, deduplication, contamination, privacy, retention,
     tokenizer, and synthetic-share gates pass;
   - teacher share and optimizer-sampling caps pass.
6. **Evaluator**
   - functional freeze identity matches;
   - semantic case identity, not only the wrapper/freeze digest, has no match in
     any prior consumption record;
   - no unchanged deterministic generator/input pair is relabeled fresh;
   - forbidden-field runtime enforcement and independent integrity checks pass;
   - learned generation, repair, rating, judging, and adjudication have no
     project-selected generated-token quality cap;
   - completion is parser/artifact complete or model EOS after exact prompt
     tokenization, and model-context exhaustion invalidates the observation.
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
python3 scripts/pre_long_run_acceleration_residual_audit.py
python3 scripts/pre_long_run_independent_readiness_audit.py
python3 scripts/pre_long_run_replacement_freeze.py
python3 scripts/roadmap_implementation_gate.py --gate --require-pre-training-ready
```

The original `reports/pretraining_architecture_freeze_package.json` binds the
historical step-3480 transaction and must not be regenerated against the
evolved step-11416 plan. The three pre-long-run commands above are the current
content-addressed readiness chain.

Then verify the repository and machine-authority state:

```bash
git status --short
python3 scripts/neural_seed_training_campaign.py
python3 scripts/neural_seed_d2_autonomous_evaluation_controller.py
```

The independent audit is currently GREEN after recomputing the 160 current
model-visible prompts against the content-addressed consumed v8 packet with
zero exact or whitespace/case-normalized overlap. The replacement package
still fails closed on `source_binding` because the maintenance tree is dirty.
The optional `runtime/control/neural_seed_yield_after_segment` file is an
emergency stop request and is normally absent; it is not approval authority. A
final package generated from a clean post-maintenance commit can authorize the
registered machine-predicate controllers to decide whether training or the
one-shot D2 evaluation may acquire its exclusive lease. Corben is not a gate.

## Transactional Qualification

When every machine predicate passes and no competing accelerator job is active,
run the registered campaign controller. It rechecks resource and lineage gates
between atomic checkpoint transactions and requires no user or operator
approval:

```bash
python3 scripts/neural_seed_autonomous_launch_controller.py --execute
```

Do not invoke the child trainer directly. Reinvoke the autonomous controller
after each committed segment; it reacquires source, resource, emergency-yield,
exclusive-lease, rollback, and lineage predicates each time. Its 64-step child
segment is the qualified atomic checkpoint transaction, not an output-token or
quality cap.

After the segment:

- verify optimizer positions rather than only nominal steps;
- rehash model, optimizer, RNG, cursor, and receipt;
- verify the appended lineage manifest;
- inspect host/swap/thermal/write evidence;
- confirm both exact contract and semantic case identities remain unconsumed;
- confirm no emergency yield request or competing accelerator lease appeared.

## Full Campaign Order

After the transactional qualification:

1. complete the modular shared-trunk candidate;
2. train the dense-active-parameter control;
3. train the dense-total-parameter control;
4. verify matched raw data, compute, tuning, inference, verifier, and total
   system cost;
5. after an explicit evaluator-integrity repair, consume one independently
   proven fresh private functional evaluation once;
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
