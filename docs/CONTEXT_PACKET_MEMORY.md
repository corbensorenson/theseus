# Context And Virtual Context Memory

Theseus now treats the old context packet ledger as the ingest adapter and
Virtual Context Memory v1 as the durable governed memory substrate.

## Ingest Layer

`scripts/context_packet_ledger.py` still collects local reports and JSONL
events into compact packets. It scores conclusions, actions, benchmark
metadata, residual reports, teacher metadata, routing traces, daemon events,
knowledge-source gates, and resource state.

Run:

```bash
python3 scripts/context_packet_ledger.py --ingest-reports --compact --out reports/context_packet_ledger.json
```

Primary artifacts:

- `reports/context_packets.jsonl`: append-only packet stream.
- `reports/context_packet_ledger.json`: active packets, summaries, drop
  candidates, and compact context view.

## VCM Layer

`scripts/virtual_context_memory.py` upgrades packets, redacted dogfood usage
events, and key project docs into typed semantic pages with stable `vcm://`
addresses. It also writes the local event log, graph manifest, transaction
ledger, snapshots, compiled context, query index, training-admission audit, and
VCM-Bench result.

Every page carries:

- page type and execution class;
- authoritative source hash and provenance role;
- L0-L5 representations, where L5 is a raw-source/hash reference rather than
  raw text copied into active context;
- compression certificates with declared loss and fallback path;
- governance labels for purpose, taint, prefetch, sharing, and training use;
- importance/risk vectors;
- explicit page-fault metadata when exactness, freshness, capability, detail,
  or capacity is missing.

Run:

```bash
python3 scripts/virtual_context_memory.py --task "Project Theseus autonomy context and memory compilation"
```

Useful local control-plane commands:

```bash
python3 scripts/virtual_context_memory.py status
python3 scripts/virtual_context_memory.py query --query "Virtual Context Memory" --limit 5
python3 scripts/virtual_context_memory.py explain --explain-address vcm://theseus/...
python3 scripts/virtual_context_memory.py record-usage --usage-kind accepted --usage-label daily_use_event --usage-summary "redacted local summary"
```

Primary artifacts:

- `reports/virtual_context_memory_events.jsonl`
- `reports/virtual_context_memory_pages.jsonl`
- `reports/virtual_context_memory_ledger.json`
- `reports/virtual_context_memory_graph.json`
- `reports/virtual_context_memory_transactions.jsonl`
- `reports/virtual_context_memory_snapshots.json`
- `reports/virtual_context_compiled_context.json`
- `reports/virtual_context_memory_bench.json`
- `reports/virtual_context_memory_bench.md`
- `reports/virtual_context_memory_index.json`
- `reports/virtual_context_memory_training_admission.json`
- `reports/virtual_context_memory_consumer_audit.json`
- `reports/virtual_context_memory_status.json`
- `reports/virtual_context_memory_usage_events.jsonl`
- `reports/virtual_context_memory_probe.json`
- `reports/virtual_context_memory_probe.md`

## Public Memory Calibration

Public memory/context benchmarks are calibration-only. Prompt/context/answer
payloads are staged only under ignored quarantine, while reports commit only
hashes, aggregate metrics, run-once ledger entries, and private-only residual
repair targets.

Current prompt-level source-of-truth artifacts:

- `reports/vcm_public_memory_prompt_calibration.json`
- `reports/vcm_public_memory_prompt_calibration.md`
- `reports/vcm_public_memory_prompt_calibration_ledger.jsonl`
- `reports/vcm_public_memory_readiness_audit.json`
- `reports/vcm_release_conformance_audit.json`
- `reports/vcm_public_memory_private_residual_repair.json`
- `reports/vcm_longmemeval_private_residual_curriculum.json`
- `reports/vcm_evidence_gauntlet.json`
- `reports/vcm_proof_card.md`
- `reports/vcm_hard_memory_private_analogues.json`
- `reports/vcm_hard_memory_benchmark_readiness.json`

The long-context ladder slice
`vcm_public_memory_prompt_slice_2026_06_19_long_context_ladder_1000_fresh`
scored 1000 prompt-level rows across RULER, BABILong, and LongMemEval with no
public training rows, no teacher solving calls, no external inference, and no
fallback returns. VCM-on scored `0.807` versus flat-tail `0.397` and best
non-VCM `0.710`. The slice includes source-context buckets around 8k, 32k, and
128k-plus token-equivalent lengths.

The current governed repair confirmation is
`vcm_public_memory_prompt_slice_2026_06_19_lme_extraction_repair_confirm_600`.
It scored 600 prompt-level rows with 200 LongMemEval rows and zero overlap
against the prior 1000-row ladder manifest. Overall VCM-on scored `0.696667`
versus flat-tail `0.331667`, with VCM-over-best-non-VCM delta `+0.086667`.
LongMemEval improved from `0.055` to `0.10`, beat flat-tail `0.02`, and tied
the best single non-VCM baseline at `0.10`. The remaining public LongMemEval
wall is not admissibility; it is semantic exact-answer quality, with private-only
repair targets for query decomposition, answer-span compaction,
structured/recency fusion, evidence recall, and abstention thresholding.

The latest governed hard-public prompt confirmation is
`vcm_public_memory_prompt_slice_2026_06_19_five_family_hard_public_memory_2196`.
It scored a fresh exact-run slice of 2196 rows across five admitted public
families: RULER, BABILong, InfiniteBench, NeedleBench/OpenCompass, and
LongBench v2. The item manifest hash is
`sha256:ba0d830ad37d3b36c68d666b854e62ad6ff79cc379e16ff1a5f856b956bd43a6`,
with zero forbidden overlap against every recorded prior prompt slice. VCM-on
scored `0.952641` versus flat-tail `0.179417` and best non-VCM `0.908925`;
VCM-over-best delta was `+0.043716`, with `1700` VCM-only wins, `2` off-only
wins, and no public training rows, teacher calls, external inference, or
fallback returns. The slice covers `lt_8k`, `8k_to_32k`, `32k_to_128k`, and
`128k_plus` buckets, with max source context `607012` token-equivalent. This is
strong positive-transfer evidence on apples-to-apples prompt-level public
memory retrieval. The honest weak points are LongBench v2 semantic choice
evidence (`0.0625` VCM-on on the admitted 16-row slice), NeedleBench answer
formatting/evidence extraction (`0.0` across deterministic systems), and
LongMemEval-V2 text/evaluator staging.

The private LongMemEval residual curriculum now covers those repair targets
without public prompts, contexts, answers, traces, tests, solutions, or answer
templates. `reports/vcm_longmemeval_private_residual_curriculum.json` is
`GREEN` on 180 private LongMemEval-style rows: VCM-on pass rate `0.983333`,
best single non-VCM `0.811111`, VCM-over-best-non-VCM delta `+0.172222`,
minimum major question-type pass rate `0.833333`, evidence recall `0.883333`,
and abstention precision/recall `1.0`/`1.0`. No teacher solving, external
inference, fallback returns, public payload loading, or public training rows are
used. This justifies proposing one future exact-once public confirmation, but it
does not itself update the public LongMemEval score.

The broad private VCM evidence gauntlet is
`reports/vcm_evidence_gauntlet.json`. It scores 1200 private/local cases across
LongMemEval-style semantic memory, RULER-style needle retrieval,
BABILong-style state tracking, and file/task memory. Current state is `GREEN`:
VCM-on pass rate `0.990833`, best single non-VCM baseline `0.809167`,
VCM-over-best-non-VCM delta `+0.181666`, minimum major-family pass rate
`0.979167`, answerable evidence recall `0.99009`, abstention precision/recall
`1.0`/`1.0`, no fallback returns, no teacher or external inference calls, and
zero public payload or training-row counters. It reports wins on file/task
memory and LongMemEval-style semantic memory, ties on BABILong and RULER where
deterministic non-VCM baselines are already perfect, and no losing family. The
report prepares a public confirmation manifest proposal but does not run public
calibration automatically.

The hard-memory VCM analogue gauntlet is
`reports/vcm_hard_memory_private_analogues.json`. It mirrors harder public
memory/long-context families without using public payloads: NoLiMa-style
lexical disconnect, Michelangelo/LSQ-style latent structure, LV-Eval-style
confusing facts, LOFT-style structured memory, LongBench v2 multi-document
bridging, InfiniteBench long dependency, MTRAG/MTRAG-UN multi-turn answerability,
FACTS-style grounding, and LoCoMo-Plus-style cognitive memory. Current state is
`GREEN`: 1000 private rows, 10 families, 4 length buckets, VCM-on pass rate
`0.979`, best single non-VCM `0.804`, delta `+0.175`, minimum family pass rate
`0.91`, evidence recall `0.952778`, abstention precision/recall `1.0`/`1.0`,
fallback returns `0`, teacher/external inference `0`, and public payload/training
counters `0`. The repair added alias/cue bridge recovery, answer-shape compaction
for key/result/mode spans, stale/confusing answer demotion, long-dependency
terminal answer selection, and explicit unknown-context abstention.

The hard public benchmark source audit is
`reports/vcm_hard_memory_benchmark_readiness.json`. Current state is `YELLOW`
by design: the private hard-memory target is met and public prompt rows are now
`2000/2000`, but the public exact-run evidence is limited to currently
prompt-ready local sources. Several candidate sources still require admission
before scoring.
NoLiMa and LoCoMo are blocked by non-commercial terms, FACTS Grounding is
blocked by model-judge/private-leaderboard scoring, and Michelangelo/LSQ, LOFT,
MTRAG/MTRAG-UN, and LoCoMo-Plus still need official source/license/evaluator
review. Do not substitute private analogues for public scores.

## Hard Invariants

- Public benchmark metadata is calibration/audit data only. It is not training
  pressure.
- Teacher metadata stays provisional/data-only unless the governed distillation
  gate accepts it.
- Tainted external text cannot promote itself into an instruction lane.
- Lossy summaries cannot satisfy exact quotation or source-replay demands.
- Stale state claims fault instead of silently becoming current truth.
- Prefetch enters non-influential staging before any model-visible promotion.
- Transactions roll back to a private branch on hard verifier/governance
  failure.
- Deletion/tombstone events trigger descendant and cache/index invalidation.
- Deleted or invalidated pages fault and cannot become model-visible context.
- Memory-derived private training rows must pass the VCM training-admission
  bridge: source hash, event provenance, taint state, public-calibration
  quarantine, teacher boundary, and deletion closure must all be clean.
- Dogfood usage events are local-only and redaction-safe by default. Raw usage
  text is not stored; the lane records hashes, labels, artifact references, and
  purpose limits.
- The VCM refresh uses no external inference and does not run public
  calibration.
- Fallback returns are forbidden and counted by the probe.

## Autonomy Integration

Each autonomy cycle refreshes context packets after metric history and then,
when `configs/autonomy_policy.json` has `virtual_context_memory.enabled=true`,
runs the VCM refresh. The refresh is local-only and writes a strict probe report
so memory quality is visible to the control plane without blocking unrelated
guarded work.

The old packet ledger remains useful as evidence ingestion. VCM is the active
contract for deciding what becomes working context, in which representation,
under which governance, with which snapshot, and with which safe fault path.
Watchdog, Hive/operator status, and dashboard summaries expose VCM freshness,
faults, graph conflicts, training-admission state, latest snapshot, and
recommended repairs.

`scripts/vcm_task_context_bridge.py` is the task-facing integration layer. It
compiles VCM into explicit context views for operator chat, autonomy
governance, code/training, public calibration review, teacher governance,
Hive/storage routing, voice/spatial operation, runtime/MLX/Metal routing, and
documentation/project-state work. The bridge writes
`reports/vcm_task_context_bridge.json` and `reports/vcm_task_contexts.json`.
Current state is `GREEN`: 9/9 task families are ready, 7/7 high-priority
families are ready, 45 unique VCM pages are selected across task views, public
training rows are `0`, external inference calls are `0`, teacher solving calls
are `0`, and fallback returns are `0`. Autonomy launch readiness now treats this
bridge as a blocker gate, and VCM-Governed release conformance includes
`task_context_bridge_clean`.

VCM v1 does not claim native model-runtime KV-cache paging. It records L5
source references and cache invalidation keys now; actual native/runtime paging
remains future work.
