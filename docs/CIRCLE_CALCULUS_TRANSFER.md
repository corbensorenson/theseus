# Circle Calculus Transfer Lane

Status: local integration note. This is not model-quality, promotion, or ASI evidence.

## Purpose

Circle Calculus exports public-safe AI contract fixtures that Theseus-Hive can consume as private experiment configuration. The local bridge is:

```text
Circle finite fixtures
  -> reports/circle_ai_contract_consumer.json
  -> private Theseus-Hive benchmark design
  -> only then quality/runtime/memory/transfer claims
```

The consumer is report-only. It does not call external or local model inference, mutate training data, write promotion evidence, or export private Theseus-Hive data back into Circle.

## Commands

From the Theseus-Hive repo:

```bash
python scripts/circle_ai_contract_consumer.py \
  --contracts "../circle math/site/data/generated/theseus_hive_ai_contracts.json"
```

Self-test the consumer without needing the Circle repo:

```bash
python scripts/circle_ai_contract_consumer.py --self-test
```

Run the named structural smoke workloads:

```bash
python scripts/circle_ai_private_workload_smoke.py \
  --contracts "../circle math/site/data/generated/theseus_hive_ai_contracts.json"
```

Self-test the smoke workload layer:

```bash
python scripts/circle_ai_private_workload_smoke.py --self-test
```

Run the deterministic proxy benchmark layer:

```bash
python scripts/circle_ai_private_proxy_benchmark.py \
  --contracts "../circle math/site/data/generated/theseus_hive_ai_contracts.json"
```

Self-test the proxy benchmark layer:

```bash
python scripts/circle_ai_private_proxy_benchmark.py --self-test
```

Default outputs:

```text
reports/circle_ai_contract_consumer.json
reports/circle_ai_contract_consumer.md
reports/circle_ai_private_workload_smoke.json
reports/circle_ai_private_workload_smoke.md
reports/circle_ai_private_proxy_benchmark.json
reports/circle_ai_private_proxy_benchmark.md
```

## Contract Families

The first Circle pack covers:

- recurrence schedules for looped/recursive work-budget diagnostics;
- strided candidate fanout for deterministic coverage comparisons;
- cyclic memory residue plus winding for alias visibility;
- MultiCoil phase features for private state-sequence feature ablations;
- circulant and block-cyclic mixers for parameter-accounting baselines;
- seed-rule exact regeneration for CGS/tool-card provenance checks.

## Claim Boundary

The consumer separates these axes for every contract:

```text
quality
runtime
memory
parameter_count
interpretability
transfer
failure_cases
```

Most axes are intentionally marked `not_measured` or `planned_private_eval` until a named workload, ordinary baseline, negative control, metric, script, and report exist.

The smoke workload report gives each contract family a named deterministic workload slot:

```text
circle_recurrence_budget_trace_smoke
circle_strided_candidate_coverage_smoke
circle_memory_alias_visibility_smoke
circle_phase_feature_invariance_smoke
circle_mixer_parameter_accounting_smoke
circle_seed_rule_regeneration_smoke
```

Those names are structural smoke checks only. They should become real private benchmarks only after actual Theseus-Hive workload data, ordinary baselines, and quality/runtime/memory metrics are attached.

The proxy benchmark report goes one step further by measuring deterministic baseline metrics for the same six families:

```text
circle_recurrence_budget_proxy_v1
circle_candidate_fanout_proxy_v1
circle_memory_alias_proxy_v1
circle_phase_feature_proxy_v1
circle_mixer_parameter_proxy_v1
circle_seed_rule_regeneration_proxy_v1
```

These proxy results are benchmark-design evidence only. They still do not train or score a learned model, do not use public calibration, and do not prove Theseus-Hive capability gains.

Allowed claim:

```text
Circle Calculus supplied finite, deterministic fixtures that can configure private Theseus-Hive AI experiments.
```

Not allowed yet:

```text
Circle Calculus improved Theseus-Hive quality, reasoning, speed, context length, public transfer, or ASI progress.
```

## Next Private Benchmark Attachments

1. Attach `recurrence_schedule` to a looped-model or work-budget smoke workload.
2. Attach `strided_candidate_fanout` to candidate generation beside sequential, random, round-robin, and local-window controls.
3. Attach `cyclic_memory_residue_winding` to context-packet traces and measure alias visibility.
4. Attach `multicoil_phase_feature` to Code LM state-sequence features with wrong-period and no-phase controls.
5. Attach `circulant_block_cyclic_mixer` to a route/ranker microbench with dense and low-rank controls.
6. Attach `seed_rule_exact_regeneration` to a repeated workflow/tool-card artifact and record exact regeneration cost.
