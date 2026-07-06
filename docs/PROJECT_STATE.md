# Project State

Last consolidated: 2026-07-06 local / 2026-07-06 UTC.

This is the current-wall page for Project Theseus. Historical state was moved
to `docs/archive/PROJECT_STATE_2026_06_22_pre_reality_harness.md`.

## Current Wall

Roadmap implementation state is now governed by
`configs/roadmap_implementation_matrix.json` and checked with
`python3 scripts/roadmap_implementation_gate.py --gate`. `roadmap.md` remains
the human narrative, but the matrix is the active implementation contract for
AI-book-derived phases 0-19. A phase is not complete until it has a registered
surface, abstraction, implementation binding, execution-spine hook, required
records, gates, docs, evidence, and an integration smoke. This prevents the
roadmap from becoming another prose-only claim surface while preserving the
late AI_book phases instead of deleting them for a cosmetic green state.
The current roadmap gate is `YELLOW` with `0` hard gaps: phase `0`
Repository Self-Model/Registry Discipline is `implemented`; phases `4`, `5`,
`6`, `7`, `8`, `11`, `12`, `17`, and `19` are `wired`; phases `3`, `10`,
`13`, `14`, `15`, and `16` are intentionally back to `partial` after the latest
ASI Stack/Claude book-mining pass; and phases `1`, `2`, `9`, and `18` are
externally frozen until trusted peers are reachable. This is roadmap
implementation state only; it is not a learned-generation or public transfer
claim. The active flagship core slice is `A1_claim_ledger_trace_kernel`; the
roadmap gate now requires active core slices to carry a valid current support
state and evidence refs. A1 is currently `synthetic-test-backed`; the full core
slice support summary records `A1_claim_ledger_trace_kernel=synthetic-test-backed`,
`A2_replacement_transaction_kernel=synthetic-test-backed`,
`E1_authority_scif_runtime_adapter_kernel=synthetic-test-backed`,
`B1_assisted_verified_assistant_product_lane=synthetic-test-backed`, and
`C1_correctness_rl_and_generator_survival_lane=synthetic-test-backed`.
`scripts/roadmap_implementation_gate.py` now has a strict pre-training
architecture-readiness mode:
`python3 scripts/roadmap_implementation_gate.py --gate --require-pre-training-ready`.
The normal roadmap gate remains `YELLOW` with `0` hard gaps so implementation
work can continue, but strict pre-training architecture readiness is now `RED`
with one blocker: six book-derived implementation phases are still partial.
Those partial phases are VCM transactional ABI/runtime parity (`3`), practical
neural seed survival/policy optimization/generation modes (`10`), semantic IR
localized repair (`13`), evidence hygiene/receipt faithfulness/claim revision
(`14`), procedural-memory-to-lookahead/tool lifecycle (`15`), and
verification-bandwidth/governance-tax routing (`16`). The external-frozen
phases `1`, `2`, `9`, and `18` still cite current network-doctor evidence:
`coordinator_unreachable`, `registered_peers_unreachable`,
`peer_inbound_only_outbound_blocked`, and `No route to host` for the trusted
Windows coordinator. All five pre-training book-reference core slices still
meet their current target support state, but the latest book-mining pass found
real missing mechanics, so training/public-calibration focus should not be the
primary roadmap claim until those partial phases are resolved or explicitly
falsified.
The post-readiness training/inference execution plan is now represented by
`configs/training_inference_execution_roadmap.json` and gated with
`python3 scripts/training_inference_execution_plan_gate.py --gate`. The gate
is designed to be the handoff from architecture readiness to actual work:
governed private training focus, the T2 private MLX training smoke, and local
assisted inference canaries are allowed next; longer bounded private training
waits for a clean smoke checkpoint; public calibration waits for positive
private semantic behavior plus a fresh non-consumed surface; production MLX
routing remains fail-closed on behavior quality; model-only ChatGPT-grade
serving is not claimed; Hive fleet training remains blocked while trusted
peers are unreachable. The plan hard-rejects public benchmark training,
runtime external inference, exact consumed public-surface reruns,
fallback/template/router/tool credit as learned generation, raw private user
text by default, and arbitrary remote execution.
Current result: `reports/training_inference_execution_plan_gate.json` is `RED`
with one failed check: `current_architecture_and_registry_ready`, because the
strict roadmap architecture gate is now `RED`. The T2 private MLX smoke remains
clean, but the next allowed training step is
`complete_partial_book_derived_phases_before_training_focus`, not another
automatic training rung. This is intentional: the next material project push is
to implement or falsify the missing book-derived mechanisms rather than declare
the architecture complete.
The AI_book crosswalk remains sticky by design: it currently indexes `1703`
AI_book source files and has `38` active roadmap backlog items, `0`
stale-source phase candidates, `58` public-safe evidence pointers, and `136`
active source-sync
review decisions. That keeps book-to-Theseus follow-up visible instead of
clearing it with superficial steward decisions.

The latest Claude book-mining packet is reconciled into `roadmap.md` and
`configs/roadmap_implementation_matrix.json` as planning evidence. The book's
chapter-level `Planned Codex test` lists are now treated as concrete missing
technique obligations until Theseus has registered implementations with matched
controls, negative controls, no-cheat audit, retained residuals, and explicit
non-claims. Its strongest recommendation, DPO on existing accepted/rejected
verifier pairs, is no longer a missing implementation: the shadow DPO update
ran and improved private preference-gap/loss metrics, then failed strict
private replay with `0` learned candidate rows and `0/16` behavior passes. The
current wall is therefore lower-level than offline preference optimization
(`DPO/IPO/ORPO/KTO/SimPO`), verifier-reward RL
(`GRPO/RLOO/ReMax/RLVR`), MTP, Medusa/EAGLE/speculative/LayerSkip generation,
diffusion/LLaDA sketch-first repair, or scale work: strict prompt/signature
body-token decode must emit non-fallback, non-template candidates that work
beyond the narrow simple-return replay before those methods can honestly
matter. The newest guarded/default static correction proves the simple-return
path can emit and pass `2/2` private replay candidates, but broad/private
replay still admits `0` candidates and passes `0/4`. A follow-up strict decode
hygiene pass now blocks malformed `isinstance` first-argument chains, bare
builtin type values used as runtime values, and constant-only control-flow
conditions. A follow-up generated-state return/dependency guard lets the
decoder finish model-created visible-input accumulators and blocks closing
top-level returns that ignore visible inputs. The simple-return replay remains
`GREEN`, while the broad/private return-finalizer-priority canary remains
`RED` with zero generated learned rows and noncredit `return None` baselines.
This does not weaken the ASI_book backlog; it orders it around the live
falsifying evidence: broad semantic/action body construction is still the wall,
not another narrow return-token or guard-family issue.

The Phase 14 artifact-retention budget is now a live gate rather than a TODO.
`configs/artifact_retention_budget_policy.json` defines report/checkpoint
budgets and retention classes, and
`python3 scripts/theseus_artifact_retention.py --budget-gate` reports
`GREEN`: hot reports are under the `1 GiB` cap, active index bytes are under
cap, unowned generated files are `0`, missing retention classes are `0`, and
hard gaps are `0`. The execute canary archived `386` report
snapshots through `reports/theseus_artifact_retention.json`, and
`reports/theseus_artifact_retention_replay_gate.json` replay-verified all
`386` archive pointers with `0` failures. The remaining retention issue is a
warning, not a hard gap: checkpoint bytes are still above the warning target
until current-reference-aware checkpoint compaction is implemented.

The 2026-07-06 weekly focus implementation pass is now represented by
`reports/theseus_weekly_focus_20260706.json` (`GREEN`). It refreshed the
registered assistant product-spine run and exported
`reports/theseus_public_safe_reference_trace_20260706.json` plus
`reports/theseus_book_importable_evidence_packs_20260706.json`. The gate
records `10` evidence packs, `7/7` expected-invalid receipt controls rejected,
residual conservation `GREEN`, verifier capacity `GREEN`, governance-tax
measurement present, `6` capability-claim dispositions, book-schema
conformance true, and the A1 claim-ledger trace kernel `GREEN` with
support state `synthetic-test-backed`. The A1 kernel proves required trace
records, support-state transitions, digest replay, source-to-verifier
continuity, expected-invalid controls, duplicate-family avoidance, and clean
no-cheat counters over the weekly-focus reference trace. It also preregistered
exactly one bounded correctness-in-the-loop generator experiment in
`configs/correctness_in_loop_generator_experiments.json`. This is
implementation-reference evidence for the book import path, not a model-quality,
public-benchmark, learned-generation, deployed-readiness, or ASI claim.

The A2 replacement-transaction kernel is now represented by
`reports/procedural_memory_route_adoption.json` (`GREEN`). It proves one
guarded default-route adoption through policy prechecks, independent
toolification/canary/registry/steward evaluators, regression guard, residual
escrow, rollback criteria, support-state transition, expected-invalid controls,
and clean no-cheat counters. Its support state is `synthetic-test-backed`:
`32/32` prechecks pass, `4` independent evaluators are recorded, `5/5`
expected-invalid controls are rejected, rollback guard is available, and
residual escrow is retained. The adopted route is local metadata workflow
compression only; it is not learned generation, model quality, public transfer,
external inference serving, or ASI evidence.

The E1 authority/SCIF runtime-adapter kernel is now represented by
`reports/governance_rights_receipt_suite.json` (`GREEN`). It proves one
side-effecting assistant/tool fixture through runtime adapter invocation,
authority transition/use receipts, effect receipt, rollback/no-rollback
boundary, confused-deputy denial, Digital SCIF handle proof, expected-invalid
controls, and clean no-cheat counters. Its support state is
`synthetic-test-backed`: the fixture records runtime adapter, authority
transition/use, `11` effect receipts, SCIF handle with raw secret absent,
confused-deputy denial, rollback/no-rollback boundary, and `6/6`
expected-invalid controls rejected. This is a reference fixture for the
authority membrane, not a claim that every runtime route is fully deployed under
the E1 adapter.

The B1 assisted verified assistant product lane is now represented by
`reports/theseus_assistant_product_lane_gate.json` (`GREEN`). It verifies the
existing assistant runtime surface across `4/4` route cases,
CLI/memory/feedback receipts, VCM readiness, deterministic tool evidence,
private verifier receipts, `80` recent metadata-only dogfood events, `66`
completed-or-accepted outcomes, `8/8` expected-invalid controls rejected, VIEA
product trace records, and zero public-training/runtime-external/fallback
counters. Its support state is `synthetic-test-backed`: current events prove
fixture/e2e product-lane wiring, not real daily usefulness. Empirical B1 support
now requires real multi-day user dogfood trace evidence with raw private text
off, verifier receipts retained, and accepted/missed/ignored/corrected/completed
outcomes spread across real use days. The gate preserves the strict code
generator semantic wall as C1 negative evidence rather than laundering product
usefulness into learned-generation capability.

The C1 correctness/RL/generator survival lane is now represented by
`reports/correctness_generator_survival_lane_gate.json` (`GREEN`). It proves one
bounded private verifier-driven learned body-token experiment under the
preregistered correctness-in-the-loop contract: `36` eligible transformer/hybrid
candidates across `8` private tasks, independent integrity recomputation,
selected compile/runtime-load rates of `0.375`, selected/pass-if-any behavior
`0.0`, no functional promotion, and zero public-training, runtime-external,
fallback, public-boundary, or integrity-mismatch counters. Its support state is
`synthetic-test-backed`: the gate now requires replay, integrity, blind-flow,
generation-mode, policy, and `9/9` expected-invalid controls while preserving
the falsifying semantic wall. This is not promotion-grade code generation or
public-transfer evidence.

The latest local MLX private rung sharpens that wall. T2 private MLX smoke
`reports/strict_generator_mlx_pretraining_probe_t2_private_smoke_20260706.json`
is `GREEN`: Apple Silicon MLX training runs on `Device(gpu, 0)`, writes
digest-bound checkpoint artifacts, consumes `601431` token positions, lowers
heldout LM loss from `8.060517` to `4.332259`, and keeps `0` public rows, `0`
external inference calls, and `0` fallback/tool credit. T3 private adaptation
`reports/strict_generator_mlx_private_adaptation_t3_semantic_private_rung_20260706.json`
is also `GREEN`: private heldout LM loss drops from `4.378092` to `2.073871`
with clean checkpoint/vocab SHA-256 digests and no public/external/fallback
counters. T4 replay
`reports/strict_generator_mlx_decode_eval_t3_semantic_private_rung_broad8_replay16_20260706.json`
is `RED`: the direct MLX decode path emitted `0` candidate rows and failed the
hard `candidate_rows_emitted` gate. The honest interpretation is therefore
`trainability/loss improvement proven, direct prompt/signature candidate
generation still not behaviorally useful`. This is why Phase 10 now prioritizes
private DPO/IPO, bounded RLVR/GRPO, MTP, GVR, lookahead/diffusion, semantic IR
localized repair, and scale/MoE ablations under no-cheat accounting rather than
more scalar CE-only tuning.
The first DPO shadow update for this lane now exists:
`reports/strict_generator_mlx_private_adaptation_dpo_pairwise_smoke_20260706.json`
is `YELLOW` with a frozen reference checkpoint, heldout LM loss improvement
from `1.419164` to `0.459953`, and a private accepted-vs-rejected
policy/reference gap delta of `+0.628971`. No-cheat counters stayed clean:
`public_training_rows=0`, `external_inference_calls=0`, and
fallback/template/router/tool credit `0`. The only failed gate is the soft
`parameter_element_update_meaningful` warning (`0.149228`). This is a real
policy-update receipt, but it is not a behavior-lift claim, not default
routing, not public transfer, and not learned-generation promotion. The next
Phase 10 proof is private decode/candidate-integrity/verifier replay of this
DPO checkpoint against the pre-update checkpoint.
That proof is now negative:
`reports/strict_generator_mlx_decode_eval_dpo_pairwise_smoke_broad8_replay16_20260706.json`
is `RED` under the same bounded `both` split, `8+8` private rows, top-k `8`,
plan-prefix/source-condition/loop-expression guard profile as the pre-DPO
T3 replay. It emitted `0` learned candidate rows, scored `0/16` private
heldout behavior passes, and kept `public_training_rows=0`,
`external_inference_calls=0`, and fallback/tool/template learned-generation
credit `0`. Standalone candidate integrity
`reports/candidate_integrity_strict_generator_mlx_decode_eval_dpo_pairwise_smoke_broad8_replay16_20260706.json`
is `YELLOW` because all `16` JSONL rows are `fallback_or_template`
baseline rows and `0` are independently verified learned candidates. Blind
information-flow audit
`reports/blind_information_flow_audit_strict_generator_mlx_decode_eval_dpo_pairwise_smoke_broad8_replay16_20260706.json`
is `GREEN` with zero invalid claims or static flow violations. The current
Phase 10 wall is therefore not "scale DPO"; it is direct learned body emission:
produce non-fallback, non-template, top-level-return candidates before another
preference/RL scaling step.
The first direct syntax-pathology cleanup after that wall is now implemented in
`scripts/neural_seed_token_decoder_support.py`: the strict body-token policy
blocks task-blind malformed continuations that dominated failed beams
(`data in data in data`, augmented assignment inside conditions, uncalled
method-attribute chains such as `data.get.strip`, excessive same-line
subscripts, long flat boolean chains, comment tokens, and builtin type objects
inside returns like `return len(list)`). The bounded canaries
`reports/strict_generator_mlx_decode_eval_dpo_pairwise_syntax_pathology_guard_broad2_replay4_v2_20260706.json`
and
`reports/strict_generator_mlx_decode_eval_dpo_pairwise_syntax_pathology_guard_broad2_replay4_v3_20260706.json`
still exit `RED` with `candidate_rows_emitted=0`, `0/4` private behavior
passes, and clean no-cheat counters, but runtime drops from the first
syntax-pathology canary's `74283` ms to about `21` seconds because malformed
beams stop earlier. This is decode-hygiene and runtime-pathology evidence
only. It does not solve direct learned emission, does not support a behavior
claim, and should not become another guard-churn lane. The next real Phase 10
patch should be a stronger trainable state-transition/AST/body-token head or
objective that emits valid update/finalizer/top-level-return structure before
DPO/GRPO/MTP/diffusion/scale work resumes.
That first trainable objective patch is now present in
`scripts/strict_generator_mlx_adaptation_weights.py` and
`scripts/strict_generator_mlx_private_adaptation.py` as
`strict_direct_body_emission_path_v1`. It adds private-only direct body-emission
path weighting over admitted target AST spans: top-level state bindings, branch
guards, loop headers, loop body state transitions, local-state returns, and
nontrivial return expressions. The bounded smoke
`reports/strict_generator_mlx_private_adaptation_direct_body_emission_path_smoke_20260706.json`
is `YELLOW`: direct-body weighting matched `128/128` private rows and `8054`
token positions, semantic-plan and source-contrastive heldout metrics improved,
heldout LM loss dropped from `1.766813` to `1.467117`, and no-cheat counters
remained clean (`0` public rows, `0` external inference calls, `0` fallback/
template/router/tool credit). The only failed gate is the existing soft
parameter-element update warning (`0.161285`). The strict replay
`reports/strict_generator_mlx_decode_eval_direct_body_emission_path_broad4_replay8_20260706.json`
is still `RED`: it emitted `0` accepted learned candidate rows and `0/8`
behavior passes, while verifier diagnostics show `4` generated attempts per
split reached runtime load before final admission rejected them. The candidate
JSONL contains only `8` private-baseline `return None` rows, correctly excluded
from learned-generation credit. The current wall has therefore narrowed again:
target-side body-emission supervision is now active, but decode still needs a
trainable local-return/finalizer continuation mechanism that produces
admissible non-fallback candidates under the same prompt/signature-only audit.
The first decoder-interface patch for that wall is now in
`scripts/strict_generator_mlx_decode_plans.py`,
`scripts/strict_generator_mlx_decode_eval.py`, and
`scripts/neural_seed_token_decoder_support.py`: it tightens strict body-token
legality around malformed bitwise/walrus/set-literal/`not` chains and exposes a
task-blind local-return continuation choice for already-bound parameter-derived
locals. The syntax-only private train-replay canary
`reports/strict_generator_mlx_decode_eval_local_return_continuation_syntax_canary_20260706.json`
is `YELLOW`, emitting `8` generated rows with `8/8` independently integrity
verified and `0` fallback returns, `0` public rows, and `0` external inference
calls. That is syntax-emission evidence only: strict nontrivial/top-level
semantic canaries
`reports/strict_generator_mlx_decode_eval_local_return_continuation_train_replay2_20260706.json`
and
`reports/strict_generator_mlx_decode_eval_local_return_continuation_broad2_replay4_20260706.json`
remain `RED` with `0` admitted learned rows and `0` behavior passes. The wall is
now precise: learned decode can emit some runtime-loadable body-token syntax,
but it still cannot reliably emit promotion-grade nontrivial top-level-return
semantics from prompt/signature alone.
The next static-guard correction in
`scripts/neural_seed_decode_static_guard.py` accepts the valid guarded/default
shape where the nontrivial return is inside the branch and the top-level return
is the fallthrough default, while still rejecting default-only and nested-only
bodies. The narrow private train-replay canary
`reports/strict_generator_mlx_decode_eval_guarded_default_static_relax_train_replay2_20260706.json`
is now `GREEN`: `2` emitted transformer/hybrid rows, `2/2` intended-behavior
passes, `2/2` integrity verified, nontrivial-return rate `1.0`, and clean
no-cheat counters (`0` public rows, `0` external inference, `0` fallback
returns). The broader canary
`reports/strict_generator_mlx_decode_eval_guarded_default_static_relax_broad2_replay4_20260706.json`
still exits `RED` with `0` admitted candidates and `0/4` behavior passes; its
top beams are malformed long expression chains. The wall has therefore moved
from "strict simple-return replay starves" to "broad prompt/signature semantic
and expression construction still fails."
A follow-up task-blind decode hygiene pass in
`scripts/strict_generator_mlx_decode_guards.py` and
`scripts/neural_seed_token_decoder_support.py` blocks three more malformed
families that dominated those broad beams: runaway `isinstance((data) and ...`
first-argument chains, bare builtin type names used where runtime values are
required such as `max(list)`, and constant/builtin-only branch conditions such
as `if -1:`. The narrow canary
`reports/strict_generator_mlx_decode_eval_condition_runtime_guard_train_replay2_20260706.json`
remains `GREEN` with `2` generated learned rows and `2/2` private train-replay
passes. The paired broad canary
`reports/strict_generator_mlx_decode_eval_condition_runtime_guard_broad2_replay4_20260706.json`
is still `RED`: `generated_candidate_rows=0`, `0/4` behavior passes, and the
candidate JSONL contains only noncredit `return None` baselines. This is
decode-hygiene evidence only. It removes a few invalid beam attractors, but it
does not solve broad learned generation and should not become another
guard-churn lane.

The execution-spine record contract is now shared in
`configs/viea_spine_record_contracts.json` and checked with
`python3 scripts/viea_spine_record_gate.py --gate`. Current assistant runtime
traces, planner nested ASI-stack records, deterministic tool evidence, execution
spine runtime records, Hive scheduler route records, Hive task-submission
execution receipt smoke records, candidate-integrity audit records,
private-verifier cascade records, and procedural-memory canary execution
records all normalize to the same VIEA record families
with zero public training rows, zero runtime external inference, and zero
fallback returns. This is implementation-cohesion evidence, not
learned-generation promotion evidence.
The same gate now also materializes `reports/viea_spine_materialized_view.json`,
which normalizes assistant, planner, deterministic-tool, and execution-spine
records into shared claim/proof/evidence, semantic IR, simulation-contract,
governance-right, constitutional-predicate, artifact, authority, route, and
failure-boundary groups. The latest spine gate is `GREEN` across `31/31`
profiles with `2227` materialized records, `184` claim/proof entries, `8`
compression records, and `1` defeater record.
The Hive scheduler route-validator bootstrap cycle is now explicit and
resolved: the scheduler may bootstrap only when required route-validator groups
are present and no no-cheat counters fault, then it reruns against the green
materialized view and records a ready route-validator receipt.
`reports/hive_scheduler.json` now contributes `hive_scheduler_route_records_v1`
as a first-class producer profile: its dry-plan scheduler report emitted `100`
route records across `10` placements plus a non-executing task-submission
receipt schema smoke with `10` execution records, covering authority, adapter,
budget, costed route, generation-mode, failure-boundary, artifact, and
claim/evidence-transition records. The receipt smoke proves record shape without
submitting arbitrary remote work; the next Hive proof is a real bounded
registered task submission when a trusted peer is reachable.
`reports/candidate_integrity_audit.json` now
contributes `candidate_integrity_producer_v1` as a first-class producer profile
too: it emits claim/proof, authority-use, generation-mode, failure-boundary,
artifact, and evidence-transition records over `456` audited candidates, `176`
independently integrity-verified learned full-body-token candidates, and `0`
integrity mismatches. `reports/private_verifier_spine_smoke.json` now
contributes `private_verifier_spine_v1` as a first-class producer profile from
the existing `scripts/code_lm_private_verifier.py` surface: it emits
claim/proof, authority transition/use, runtime-adapter, resource-budget,
generation-mode, failure-boundary, governed `context_transaction`,
`context_adequacy`, artifact, and evidence-transition records for a tiny
private verifier cascade smoke. The verifier smoke is `GREEN` with
`vcm_context_governor_ready=true` and
`vcm_context_adequacy_state=governed_sufficient_for_verification`. This makes
phases 1, 2, 3, 13, 14, 17, and 18 more tightly wired.
`reports/neural_seed_strict_generator_fanout_receipt.json` now contributes
`strict_generator_fanout_receipt_v1` as a first-class producer profile too. It
replays the current strict body-token candidate manifest through the private
verifier after independent candidate-integrity recomputation: `93`
integrity-verified learned full-body candidates, `22` private eval tasks,
syntax-valid rate `0.684211`, runtime-load task rate `1.0`, and intended
behavior pass rate `0.227273`. This is the Phase 10 wall in sharper form:
strict learned candidates now mostly parse/load, but semantic behavior still
needs repair. The receipt now includes
`project_theseus_strict_generator_blind_semantic_residual_diagnosis_v1`: it
uses only prompt feature tags, candidate AST/static features, rank metadata,
and verifier stage outcomes, explicitly excluding tests, hidden tests,
solutions, answer labels, source IDs, and decoder-contract target fields. It
finds `17` runtime-loaded failed tasks, `0` no-load tasks, and dominant failed
issue labels `blind_static_shape_plausible`,
`prompt_implies_branching_but_no_branch`,
`prompt_implies_structured_output_but_no_collection_construction`, and
`prompt_implies_string_processing_but_no_string_ops`. The same receipt now
includes `project_theseus_strict_generator_blind_selector_ablation_v1`: baseline
model rank, prompt/AST-only blind rank, and full-pool pass-if-any oracle all
remain at `5/22` behavior passes (`baseline_to_oracle_pass_delta=0`), even
though the blind ranker changes `8` top-1 selections and `13` top-k selections.
That makes the wall candidate-pool semantic quality, not just selector quality
or fanout width. No public training rows, runtime external inference, fallback
returns, fixed renderers, routers, tools, or body templates are credited as
learned generation.
`reports/correctness_generator_survival_lane_gate.json` now binds that wall to
the book-reference C1 slice. The stricter assistant replay fixture filters the
same generator family to integrity-verified transformer/hybrid body-token
candidates only; it has `36` eligible candidates over `8` private tasks, but
selected/pass-if-any/functional-promotion rates remain `0.0`. That is a
reference falsification fixture, not a capability win, and the next private
repair target stays semantic candidate construction before any public
calibration spend.
`scripts/strict_generator_mlx_private_adaptation.py` now has a named private
repair profile, `strict_full_body_semantic_construction_v1`, that composes the
existing source-contrastive, semantic-plan, semantic-slot, loop-update,
expression-synthesis, plan-conditioned body, update-contract, return-expression,
and primary-dataflow losses without adding templates, tools, teacher output, or
candidate credit. The profile correctly fails RED against a plain body-token
checkpoint because semantic-plan/slot targets are absent. Against the
semantic-slot MLX checkpoint
`reports/strict_generator_mlx_pretraining_visible_ops_loop_smoke_v1.json`, the
private smoke
`reports/strict_generator_mlx_private_adaptation_semantic_construction_profile_semantic_slot_private_smoke_v1.json`
is `GREEN`: semantic-plan loss improves, source-contrastive gap improves,
visible-operation plan rows boost (`37`), semantic-slot prefix weighting
matches `1012` positions, loop-semantic weighting matches `1290`, expression
synthesis weighting matches `1296`, plan-conditioned body weighting matches
`1422`, and update-contract weighting matches `576`, all with `0` public
training rows and `0` external inference. The follow-up decode smoke
`reports/strict_generator_mlx_decode_eval_semantic_construction_profile_broad4_v1.json`
is still `RED`: broad-private pass remains `0/4`, nontrivial-return rate is
`0.0`, and generated learned-prefix beams starve inside repeated loop-update
calls with `inside_loop_without_update` and `missing_local_return` states. The
first body-transition guard follow-up,
`reports/strict_generator_mlx_decode_eval_body_transition_guard_broad4_v1.json`,
removes that emission/starvation wall on the same slice: `8` generated
transformer/hybrid rows are integrity-clean, verifier labels attach, and
zero-candidate tasks fall from `4/4` to `0/4`, still with `0` public training
rows, `0` external inference calls, and `0` fallback/template/router/tool
credit. This is not a promotion: broad-private behavior remains `0/4`,
nontrivial-return rate remains `0.0`, and the new residual labels are
`loop_without_decision_or_state_update` and `missing_semantic_update_value`.
The next Phase 10 target is therefore learned semantic update choice and
nontrivial local-return synthesis, not more generic semantic weighting.
The next private replay patch now wires independent candidate-integrity
failures into that same training path instead of creating a side lane:
`scripts/strict_generator_mlx_private_adaptation.py` recomputes candidate
integrity for failed private-train replay candidates and turns syntax,
no-function, inert, and full-body mismatch findings into private negative
replay labels. The bounded smoke
`reports/strict_generator_mlx_private_adaptation_source_condition_operation_integrity_negative_replay_smoke_v1.json`
is `GREEN`: it selects `16` failed private-train replay rows, records `4`
no-function/syntax-invalid integrity-negative rows, keeps `0` public training
rows, `0` external inference, and `0` candidate-generation credit, and runs
pairwise replay with integrity-aware token weighting. The follow-up
`reports/strict_generator_mlx_decode_eval_source_condition_operation_integrity_negative_replay_broad4_v1.json`
is still `YELLOW`: `7` generated transformer/hybrid rows, `6/7`
integrity-verified, `1` no-function/syntax mismatch, nontrivial-return rate
`0.857143`, and broad-private behavior `0/4`. This is useful Phase 10
training-path coverage and negative-evidence plumbing, not a capability win or
promotion claim.
`reports/semantic_ir_obligation_gate.json` now contributes
`semantic_ir_obligation_spine_v1`. It binds candidate-integrity,
private-verifier, and direct-generator obligations to the materialized semantic
IR view: `25` semantic atoms, `93` semantic nodes, `3/3` ready consumers, `3`
semantic-obligation records, `3` dependency edges, and `3` evidence bindings.
This is Phase 13 implementation cohesion; it is not model decoding, verifier
execution, public calibration, or learned-generation promotion evidence.
`reports/code_lm_train_once_fanout.json` now contributes
`train_once_fanout_spine_v1` as a first-class producer profile too. The
canonical train-once fanout supervisor emits `10` VIEA fanout records,
including governed `context_transaction` and `context_adequacy` records from
`reports/vcm_context_governor.json`, with
`vcm_context_governor_ready=true` and
`vcm_context_adequacy_state=governed_sufficient_for_generation_fanout`. This
is fanout traceability/context evidence only; it is not a training run, public
calibration run, or learned-generation promotion claim.
`reports/neural_seed_token_decoder_comparator.json` now contributes
`direct_generator_context_spine_v1` as a first-class producer profile. The
direct neural seed generator boundary consumes the same VCM governor receipt,
fails closed before model decode if it is missing, and emits `10` VIEA
direct-generator records including governed `context_transaction` and
`context_adequacy`. The planned comparator report is `PLANNED` with
`direct_generator_vcm_context_ready=true`; this is boundary evidence, not
training or code-generation capability evidence.
`reports/vcm_native_runtime_probe.json` now contributes
`vcm_native_runtime_spine_v1` as a first-class producer profile. The runtime
boundary emits `11` VIEA runtime records over authority, context transaction
and adequacy, runtime adapter, resource, costed route, generation mode,
failure boundary, artifact, claim, and evidence-transition families. The
current probe is `GREEN`: semantic VCM descriptor cache readiness is green,
backend-scoped CPU Transformers DynamicCache prefix/KV lifecycle is proven,
and MLX resident tensor descriptor reuse/invalidation is proven. It does not
claim model-native MLX KV/prefix parity; scheduler native KV routing for the
recommended MLX backend remains fail-closed until that exact backend has a
lifecycle proof.
`reports/vcm_context_governor.json` now also contributes
`vcm_context_abi_fixture_spine_v1` and
`vcm_context_resolver_conformance_spine_v1`. The Context ABI fixture gate is
`GREEN` with `5/5` fixtures and `40` VIEA records covering valid leased
materialization, mandatory miss typed fault, verification-inadequate rejection,
mount-policy denial, and expired-lease reuse blocking. The deployed resolver
conformance lane is also `ready`: `7/7` real semantic-address requests pass,
`3` local artifact refs are materialized, `4` typed faults are emitted for
blocked/missing/stale/unsafe requests, and `56` resolver VIEA records are
materialized without raw payload leakage. This is context ABI and resolver
evidence, not a VCM benchmark score, native KV-cache parity claim, or
learned-generation claim.
`reports/deterministic_tool_substrate.json` now contributes
`deterministic_tool_context_spine_v1` in addition to the existing deterministic
tool evidence profile. The deterministic tool path is `GREEN` with `13` local
tool cards, `15/15` verified private smoke results, `7` VCM tool context
records, `0` public training rows, `0` runtime external inference calls, and
`0` fallback returns. This is exact local tool evidence and VCM context
governance, not model-only skill evidence.
The normal plan compiler now consumes that tool report for every compiled
planning node. `reports/theseus_plan_compiler.json` is `GREEN` with `25/25`
nodes declaring tool eligibility, `25/25` nodes carrying tool receipts, `11`
required tool-eligible nodes, `14` optional tool-eligible nodes, and `71`
tool-call receipts. `plan_compiler_nested_spine_v1` now requires
`tool_call_receipt` records on planner nodes. These receipts are evidence refs
for deterministic tool availability/results only; they explicitly cannot
support learned-generation claims.
`reports/training_data_admission_v1.json` now contributes
`training_data_admission_context_spine_v1`. Training-data admission consumes
the VCM governor receipt with resolver status `ready`, keeps public benchmark
payloads and raw user text out of admitted training paths, and emits VIEA
context records for its metadata-only admission report. This is admission
governance evidence, not training or public calibration.
`reports/public_calibration_proposal_gate.json` now contributes
`public_calibration_proposal_spine_v1`. Public calibration proposals must cite
candidate integrity, the training-data firewall, alignment preflight, and the
exact public run registry state before execution can count as evidence. The
current default proposal is `GREEN` as a fail-closed refusal because
`industry_code_transfer_seed14_5x64_v1` is already consumed; fresh surfaces can
still run without calendar throttles when the same gate reports them clean.
`reports/governance_rights_receipt_suite.json` now contributes
`governance_rights_receipt_spine_v1`. The fixture gate is `GREEN` with `4/4`
governance-right fixtures covering complete audit response, justified redaction
with appeal, portable exit export, and fork denial when safety obligations
cannot transfer. It now also has `4/4` constitutional-predicate fixtures and
`4` VIEA `constitutional_predicate` records covering least-sufficient-power
route selection, predicate conflict routing, constitutional migration record
requirements, and self-modification weakening rejection.
`reports/hive_operator_governance_audit.json` now contributes
`hive_operator_governance_audit_spine_v1`. The live local operator
audit/export request is `GREEN` with `13` audit refs, `6/6` applicable local
artifact payload citations passing, `1` governance-right record, `1`
authority-use receipt, `1` failure-boundary record, `13` artifact-graph
records, `1` claim record, and `13` evidence-transition records. The endpoint
is `/api/hive/operator/governance-audit`, and `/mobile` exposes it through the
Governance Audit card. The report exports refs/hashes/citations and appeal
paths, not raw private text, secrets, public benchmark payloads, hidden tests,
or learned-generation evidence.
`reports/simulation_fidelity_receipt_suite.json` now contributes
`simulation_fidelity_receipt_spine_v1`. The gate is `GREEN` with `5/5`
fixtures plus one real bounded planning-world adapter over
`reports/theseus_plan_compiled_dags.json`: `real_planning_adapter_count=1`,
`planning_adapter_passed=true`, `6` simulation-contract records, `6`
fidelity records, `6` counterfactual traces, `6` world-adapter receipts, and
`6` failure-boundary records. These are fixture and compile-time planning
claim-boundary proofs, not institutional governance, moral correctness,
physical feasibility, benchmark transfer, live simulator evidence, deployment
evidence, or learned-generation evidence.
`reports/theseus_assistant_product_spine_smoke.json`
now provides the product-facing assistant trace: a tool-lane call with VCM
ready, deterministic tool evidence `GREEN`, private verifier receipt `GREEN`,
materialized VIEA receipt ready, dogfood event written, `16` assistant VIEA
records, and `0` public training rows/runtime external inference/fallback
returns. The next gap is replacing the receipt schema smoke with a live
reachable-peer registered-task submission proof and making normal assistant
defaults traverse this same spine.
`configs/assistant_trace_schema.json` is now the shared assistant/dogfood
outcome contract. `scripts/theseus_assistant_runtime.py` loads it from
`configs/theseus_assistant_runtime.json`, validates the five allowed outcomes
(`accepted`, `missed`, `ignored`, `corrected`, `completed`), exposes the schema
hash in the runtime summary and conversation event, and embeds the schema
receipt in the VIEA `policy_optimization_record`. The current schema smoke is
`reports/theseus_assistant_trace_schema_smoke.json`, which is `GREEN` with
schema ready, dogfood metadata written, VCM/context-governor ready, private
verifier receipt ready, and `0` public training rows/runtime external
inference/fallback returns. This is product trace cohesion, not a usefulness
training claim.
`reports/theseus_assistant_product_lane_gate.json` is now the B1 product-lane
gate (`GREEN`). It verifies `4/4` assistant route cases, CLI/memory/feedback
receipts, VCM readiness, deterministic tool evidence, private verifier
receipts, metadata-only dogfood pressure, VIEA product trace records, and zero
public-training/runtime-external/fallback counters. Its support state is
`synthetic-test-backed`: fixture and e2e wiring are proven, but empirical daily
usefulness still requires real multi-day dogfood traces. It explicitly
preserves the current strict code-generator semantic wall as C1 evidence
instead of counting product/tool assistance as learned generation.
`reports/procedural_memory_toolification.json` now contributes
`procedural_memory_toolification_spine_v1` as a first-class producer profile.
The gate is `GREEN`: it consumed `397` schema-bound assistant/dogfood trace
events, produced `10` assistant-trace procedural candidates and `17` total
procedural candidates, emitted `124` VIEA procedural tool records, and kept raw
private text, public training rows, runtime external inference, and fallback
returns at `0`. These records are candidate-only workflow-compression evidence;
they are not learned-generation claims and do not become default routes until
replay regression and registry adoption pass. The first replay fixture,
`replay.local_planning_assistant_metadata_only_v1`, now passes for
`procedural.assistant_trace.9b50fc9d7d977f46`: it verifies `55` repeated
planning-assistant traces, `1.0` success/useful rates, low risk, `E1` runtime,
VIEA binding, no residuals, no raw private text, and clean no-cheat counters.
That fixture emits one registry-gated planner canary route,
`canary.local_planning_assistant_metadata_only_v1`, with default routing still
explicitly false. `scripts/theseus_plan_compiler.py` now consumes that report
and compiles one canary-only procedural-memory goal, raising the plan compiler
to `8` compiled goals, `22` nodes, and `22` trace rows while staying `GREEN`.
`scripts/procedural_memory_canary_executor.py` now executes that route in
bounded local metadata-replay mode. The execution report is `GREEN`: it matched
the same `55` schema-bound planning-assistant events, emitted `1` route packet,
adopted `0` default routes, made `0` learned-generation claims, wrote `10`
VIEA canary-execution records, measured actual duplicate-work delta `-54`, and
measured metadata verification-cost delta `-206`. This is canary execution
evidence only; default procedural-tool adoption still requires a registry
adoption transaction and continued regression cleanliness.
`scripts/procedural_memory_route_adoption_gate.py` now provides that adoption
transaction. The adoption report is `GREEN`: it adopts `1` guarded default
route, arms `1` continued regression guard, emits `12` VIEA adoption records,
keeps learned-generation claims at `0`, and keeps public training rows, runtime
external inference, and fallback returns at `0`. `scripts/theseus_plan_compiler.py`
now consumes the adoption report and compiles `1` default-route maintenance
goal, so the planner report is `GREEN` with `9` compiled goals, `25` nodes,
`25` trace rows, `821` ASI Stack record IDs, and `72` goal governance record
groups. This is local metadata workflow compression, not model capability
evidence.
`scripts/theseus_assistant_runtime.py` now consumes the same adoption report
for planning intent through `configs/theseus_assistant_runtime.json`. The route
is no longer an implicit first-clean-route choice: the assistant must match the
adoption report's `route_binding_contract` over `surface`, `intent`,
`assistant_lane`, and `vcm_task_family`, and the hard
`procedural_default_route_binding_contract_enforced` gate must pass before the
route can attach to a runtime trace. The roadmap and trace-schema assistant
smokes are `GREEN` with `procedural_default_route_ready=true`,
`procedural_default_route_selection_matched=true`, selection mode
`route_binding_contract`, route `default.local_planning_assistant_metadata_only_v1`,
guard armed, learned-generation claims disallowed, `19` assistant VIEA records
per smoke, and zero public training rows, runtime external inference calls, or
fallback returns. Those canonical assistant traces are now part of the
`2227`-record VIEA
materialized view with `184` claim/proof entries. This is product-route consumption evidence, still not a
learned-generation or public-transfer claim.
`reports/octopus_router_report.json` now contributes
`octopus_moecot_route_spine_v1` as a first-class producer profile. The
canonical Octopus report is `GREEN`: `14` route decisions emit `251`
VIEA-normalized MoECOT route records across route decisions, specialist
semantic nodes, authority receipts, VCM context transactions and adequacy,
runtime adapters, resource budgets, costed routes, generation-mode boundaries,
failure boundaries, artifacts, claims, evidence transitions, and residuals.
The report attaches a ready VIEA route-validator receipt, a ready VCM context
governor receipt, and a ready candidate-integrity boundary; it records `0`
public training rows, `0` runtime external inference calls, `0` fallback
returns, and `0` learned-generation credit. This makes Octopus a governed
capability selector, not a learned code-generation or benchmark-performance
claim. `reports/octopus_router_head_report.json` now also contributes
`octopus_router_head_training_spine_v1`: the sparse router head trains from
`10` schema-bound real task-to-arm traces plus seeded router cases, emits `240`
wrong-labelset contrastive negatives, passes `42/42` holdout contrastive
negative checks, and keeps learned-generation credit at `0`. The rule router
remains the deterministic bootloader; this is routing-head evidence only.

The control plane now consumes the view: `reports/theseus_control_plane.json`
includes a `viea_spine_materialized_view_ready` gate, which passed with `1435`
records and `129` claim/proof entries before the Octopus route-spine producer
was added. The current shared VIEA view is `2227` records and `184`
claim/proof entries after Hive route/execution claim records,
report-evidence compression records, retention compression records, the
defeater producer contract, Octopus route-spine records, the learned router-head
training spine, and Context ABI fixture, governance-rights receipt, and
simulation-fidelity receipt records were materialized. The overall control plane remains
`RED`: the latest refresh sees `15` stale control reports, `27` blockers, and
`4` hard blockers. This is no longer because the registry, VIEA spine, or
AI_book source-sync backlog is stale; `reports/theseus_project_registry.json`
is `GREEN` with `12` routing-eligible implementations, zero routing blockers,
and zero hard governance violations. `reports/closed_loop_residual_ratchet.json` now exists and is
`YELLOW` with exactly one decision, `retry_private`; it cites broad public pass
rate `0.14375`, private repair pressure rows `445`, same-seed private semantic
lift `0.25`, and dominant residuals in `edge_case`, `type_handling`,
`local_code_generation_adapter_needed`, and `parsing`. The ASI wall governor
now points at that private-retry command instead of a missing report. Public
calibration, model growth, and candidate promotion remain blocked by the real
capability gates, not by a missing control artifact.
The first direct consumers are now wired too: `reports/hive_policy_first_scheduler.json`
is `YELLOW` with its `viea_spine_view_ready=true` receipt and only the honest
remote-CUDA/no-eligible-device warning remaining; `reports/candidate_integrity_audit.json`
is `GREEN`, records a `candidate_integrity_harness` consumer receipt, and now
also emits producer-side integrity records over claim/proof, artifact,
failure-boundary, and generation-mode groups; private verifier correctness
summaries now emit producer-side spine records through
`reports/private_verifier_spine_smoke.json`. The project registry's SCF route
validator now consumes the same materialized governance/failure/authority/resource
view before route approval; `reports/theseus_project_registry.json` records
`route_validator_viea_spine_view_ready=true` with `2227` records and `0` missing
required groups. The product-facing assistant trace and Hive scheduler route
records now both attach ready route-validator receipts, so normal route-decision
records show which governance/failure/authority/resource view approved them.
Hive artifact surfaces now cite the same spine too:
`reports/hive_installer_artifacts.json`, `reports/hive_artifact_index_smoke.json`,
`reports/hive_artifact_sync.json`, and `reports/hive_artifact_merge_summary.json`
all carry ready `viea_artifact_citation` receipts with claim-ledger,
artifact-record, and evidence-transition refs and zero public-training,
runtime-external-inference, or fallback-return counters. Those citations now use
`best_valid_support_by_artifact_family_support_state_and_no_cheat_cleanliness_v1`,
so Hive manifests prefer Hive-native claim/artifact/evidence rows when those
records exist. The current installer
manifest honestly has `artifact_count=0` because this working tree has not
rebuilt package artifacts in this slice; the citation path is ready, not a DMG
availability claim.
`reports/report_evidence_store.json` now emits `compression_record` rows for
large report snapshots, strict `compressed_artifact_record` rows, strict
`compression_receipt` rows, and `defeater_record` rows when mutable latest-view
paths are superseded. `reports/ratcheting_modular_intelligence_report.json`
also emits one conservative `compact_generative_record` for the RMI operating
map. That record now carries the AI_book-required generation, verification,
fallback, residual-burden, promotion-blocker, source-ref, evidence-ref, and
non-claim fields, and the VIEA schema gate fails if compact records omit them.
`reports/theseus_artifact_retention.json` emits the same record families during
retention selection when candidates exist. The Phase 14 execute canary
archived one generated report snapshot, left a pointer at the original path,
and `reports/theseus_artifact_retention_replay_gate.json` independently
verified pointer resolution, gzip decode, exact SHA-256 replay, JSON parsing,
and defeater reconstruction.
This is route/evidence cohesion. It is not a public-transfer result and not a
learned-generation promotion claim.

The roadmap gate now writes `reports/book_to_theseus_crosswalk.json` from the
governed matrix and refreshes it from `/Users/corbensorenson/Documents/AI_book`
source checksums. The current authored-source inventory has `1703` files; the
exact manifest hash is gate-owned in
`reports/book_to_theseus_crosswalk.json` so the human docs do not churn when the
living book changes. Generated book builds, archives, and Lean build outputs
are excluded. The crosswalk maps
all `20` roadmap phases to AI_book source basis, registry surface, abstraction,
implementation, evidence, missing items, smallest next patch, source-sync
support state, sticky stale-source backlog rows, and public-safe
Theseus-to-book evidence pointers without storing raw book text. The July 6
weekly-focus reference trace and evidence-pack exports are now imported into
AI_book through the generic reference-trace fixture
`experiments/reference_trace/fixtures/valid_theseus_weekly_focus_20260706_runtime_trace.json`
and validated by `python3 scripts/validate_reference_trace.py`, replayed
through `python3 scripts/validate_reference_trace_replay.py`, and covered by
`python3 scripts/validate_receipt_repository_audit.py` plus
`python3 scripts/validate_book.py`. The source-sync smoke passes,
public-safe evidence smoke passes, and the current crosswalk has `57`
public-safe evidence pointers, `38` active roadmap backlog rows, `136`
source-sync review decisions, `38` module source-backlog work cards, and `0`
currently stale phases. The sticky-source rule remains in force: future AI_book
authored-source changes must create carry-forward backlog rows until an exact
`source_sync_review_decision` either updates the registry-owned matrix or
records that no implementation-contract change is needed. This is Phase 19
wired evidence, not model capability evidence.

The project registry now supports manifest-owned per-report freshness policies.
Long-running immutable MLX checkpoint provenance and hash-addressed 30M-rung
decode replay reports can carry an explicit weekly freshness lease while fast
canaries/latest-health reports keep tighter freshness. This prevents wasteful
daily full reruns without making stale current-health evidence routable. The
registry gate is currently `GREEN` with `0` implementation routing blockers,
`0` hard governance violations, `12` routing-eligible implementations, `16/16`
cleanup queue items covered by active steward decisions, and `0` stale or
missing declared report outputs. A full private MLX rung replay on this Mac was
stopped after refreshing three child artifacts because it exposed a real
`mlx.core.eval`/verifier hot-loop bottleneck. The bounded canary
`reports/strict_generator_mlx_rung_decode_sweep_plan_prefix_plan_aux_30m_rungs_canary_v1.json`
is now `GREEN`: it selected `1/5` hash-addressed checkpoints, ran one
family-disjoint private row through MLX replay in `3532` ms total / `2497` ms
child decode-eval time, wrote one child decode report, reused the private split
selection through a hash receipt, and recorded zero public training rows, zero
external inference, zero template/router/tool credit, and zero integrity
mismatches. The child decode remains `YELLOW` with `0/1` passes, so this is
route-health and runtime-economics evidence only. The two-checkpoint receipt
below supersedes this one-checkpoint bottleneck note with split, top-k,
checkpoint-loader, and verifier-cache reuse evidence; neither receipt is a
capability claim.

A two-checkpoint bounded follow-up,
`reports/strict_generator_mlx_rung_decode_sweep_plan_prefix_plan_aux_30m_rungs_two_checkpoint_canary_v1.json`,
is also `GREEN`: it selected `2/5` checkpoints, reused one private
family-disjoint split receipt across both checkpoint replays, kept all no-cheat
counters at zero, and wrote child decode receipts with
`decode_topk_selection.policy = argpartition_bounded_topk_v1` and
`full_vocabulary_sort_in_hot_loop = false`. It now also proves same-vocab
checkpoint-loader reuse while still reloading every checkpoint weight file:
`vocab_read_count=1`, `vocab_cache_hit_count=1`, `model_construct_count=1`,
`model_reuse_count=1`, `checkpoint_weight_load_count=2`, and the hard
`checkpoint_loader_reloads_each_checkpoint` gate passes. It also emits the
verifier sandbox warmup rollup
`private_verifier_sandbox_warmup_accounting_rollup_v1`: `receipt_count=2`,
test-harness compile caching enabled, and
`total_test_harness_cache_hit_count=2`. The two rows still pass `0/2` private
tasks, so this remains route/runtime evidence only.

The bounded all-rung follow-through
`reports/strict_generator_mlx_rung_decode_sweep_plan_prefix_plan_aux_30m_rungs_all_rung_bounded_v1.json`
is `GREEN`: it selected all `5/5` available rungs, passed the hard
`all_available_checkpoints_selected_when_unbounded` gate, reused one
family-disjoint private split across every checkpoint, reused one same-vocab
model construction, reloaded all five checkpoint weight files, and emitted five
verifier-cache warmup receipts. Loader stats are `vocab_read_count=1`,
`vocab_cache_hit_count=4`, `model_construct_count=1`, `model_reuse_count=4`,
and `checkpoint_weight_load_count=5`; verifier warmup reports
`receipt_count=5` and `total_test_harness_cache_hit_count=5`. The all-rung
smoke also emits `project_theseus_mlx_rung_replay_resource_budget_v1`; the
no-threshold baseline passes the `resource_budget_thresholds_respected` gate.
It now also emits
`project_theseus_mlx_rung_replay_route_eligibility_v1`: the baseline is
explicitly `production_route_eligible=false` with
`route_state=observation_only_no_resource_thresholds`. It still records `0/5`
private behavior passes and one final rung with nontrivial-return rate `0.0`,
so this closes a route-health/runtime proof but does not move learned code
capability.

The strict resource-budget probe
`reports/strict_generator_mlx_rung_decode_sweep_plan_prefix_plan_aux_30m_rungs_resource_budget_probe_v1.json`
is now `GREEN` on route-resource mechanics under a 30s total budget, 10s max
child budget, and 0.2 eval rows/sec floor: `budget_ok=true`,
`max_child_decode_eval_runtime_ms=6451`, and `eval_rows_per_second=0.300084`.
The new route-eligibility receipt still fail-closes production routing with
`production_route_eligible=false` and
`route_state=fail_closed_behavior_quality_zero` because the replay remains
`0/5` on private behavior. The next Mac acceleration patch is therefore not a
latency-only patch; it is broader private replay plus semantic candidate-quality
repair under explicit resource and no-cheat thresholds before any production
MLX route claim.

The broader private MLX replay now has a two-row all-rung resource receipt,
`reports/strict_generator_mlx_rung_decode_sweep_plan_prefix_plan_aux_30m_rungs_broader_resource_probe_v2.json`.
It is `GREEN` under a 65s total budget, 25s max child budget, and 0.15 eval
rows/sec floor: `total_eval_rows=10`, `total_generated_candidate_rows=10`,
`eval_rows_per_second=0.1906`, `max_child_decode_eval_runtime_ms=16961`,
`checkpoint_weight_load_count=5`, `model_reuse_count=4`,
`total_test_harness_cache_hit_count=10`, and all public-training,
external-inference, fallback/template/router/tool-credit, and integrity-mismatch
counters remain zero. The immediately prior stricter probe
`reports/strict_generator_mlx_rung_decode_sweep_plan_prefix_plan_aux_30m_rungs_broader_resource_probe_v1.json`
correctly failed closed with `route_state=fail_closed_resource_budget` because
`rung_25000000` took `21274` ms against a 20s child budget. The current broader
receipt therefore proves resource-threshold accounting and route fail-closed
behavior on a larger private replay, but still records `0/10` behavior passes
and `production_route_eligible=false` with
`route_state=fail_closed_behavior_quality_zero`.
`reports/resource_mlx_route_readiness_gate.json` is now `GREEN` with `0` failed
checks and `0` failed expected-invalid controls, so Phase 8 is wired as a
resource-route readiness surface. This is not a production-route, model-quality,
or CUDA/MLX/Metal parity claim: production MLX routing remains disabled while
behavior is zero, parity remains unclaimed, and semantic candidate quality is
the next governed training wall rather than a roadmap-architecture blocker.

Teacher/data governance now has a durable share ledger view:
`reports/teacher_share_ledger_summary.json` is `GREEN` with
`metric_ready=true`, `19` ledger rows, `64152` accepted training rows in the
denominator, `3` accepted teacher rows, `7` verified self-generated rows, and
`teacher_share_of_accepted_training_rows=0.00004676393565282454`. It records
`10` training-time external teacher calls, but `0` runtime external inference
calls and `0` public training rows. `scripts/theseus_control_plane.py` consumes
that report through the `teacher_share_ledger_ready` gate, and
`scripts/theseus_assistant_runtime.py` consumes it through the
`teacher_share_ledger_metric_ready` hard gate in
`reports/theseus_assistant_teacher_share_smoke.json`.
`scripts/hive_node.py operator-status --out reports/hive_operator_status.json`
now exposes the same durable teacher/governance state in the operator/mobile
payload as `teacher_governance`, including governance-right fixture counts,
runtime external inference `0`, public training rows `0`, fallback returns `0`,
and `no_cheat_clean=true`. Phase 7 is wired; the remaining work is a true
multi-cycle trend direction once more teacher/self-generated cycles exist.

Theseus is not blocked by public benchmark permission or a calendar budget. It
is blocked by blind capability evidence. The previous clean private
transformer/hybrid `64/64` result is invalid for learned-code-generation or
promotion claims because the action-selector path allowed answer-identifying
metadata to influence ranking:

2026-06-23 UTC flywheel rebaseline: the governed teacher-distillation path now
admits only private execution-verified code-LM rows and rejects malformed or
generic teacher rows instead of laundering them into training. The previously
accepted generic planning rows are retained as non-training evidence and
rejected with `code_lm_task_missing`; `private_parse_filter_count_order_v1` is
rejected because its `solution_body` contained a full `def`; a later
`records`-argument row is rejected because it failed the private execution
verifier. Current durable counts are `3` manifest-admitted executable teacher
rows, `4` rejected teacher rows, `5` retained proposal rows, `5`
verified-self-generated ledger rows, `0` public training rows, and runtime
external serving remains forbidden. The teacher manifest now dedents admitted
code-LM solution bodies before they become body-token targets, so teacher rows
cannot accidentally train an artificial wrapper-indentation token stream.

Two bounded flywheel rounds have now run. Round 1 consumed the two existing
executable teacher rows and preserved the honest strict baseline: in-family
learned full-body token pass stayed `5/24 = 0.208333` and strict
family-disjoint transfer stayed `0/24` with `0` held-out-family teacher rows.
Round 2 admitted `private_state_window_relay_v1`, a verifier-accepted private
teacher row targeting stateful algorithm planning and return-shape realization,
but the strict comparator still stayed `5/24` in-family and `0/24`
family-disjoint while syntax regressed (`0.777778 -> 0.722222` SymLiquid
in-family, `0.708333 -> 0.430556` SymLiquid family-disjoint, `0.5 -> 0.486111`
transformer in-family). That row remains retained in the manifest/ledger, but
the strict comparator loader now utility-quarantines it by task id so future
student runs consume only the two non-negative admitted teacher rows until a
stronger utility gate reinstates it.

A hard variable-scope/colon pruning decoder ablation was tested and rejected:
it collapsed in-family pass to `2/24` and family-disjoint syntax to as low as
`0.0`. The narrowed repair now keeps dedent normalization and prompt/signature
visible subword features only. A later private-training-target local-name
canonicalization ablation was also rejected and disabled: it drove the main
strict verifier pass to `0/24`, dropped SymLiquid/transformer STS-on syntax to
`0.625/0.333333`, and still left family-disjoint transfer at `0/24`.
Restoring the non-normalized target path restores the prior honest baseline.
Candidate integrity is GREEN with `0` mismatches; blind information-flow is
GREEN and now audits the strict comparator sources, config, candidate manifest,
and strict report by default. This moved governance and flywheel honesty, not
broad transfer. The current wall is still prompt/signature body-token semantic
generalization under held-out families, not teacher admission, ledgering, or
syntax-only validity.

`reports/candidate_promotion_gate.json` now requires the candidate-integrity
VIEA consumer receipt before a promotion report can pass. The current
promotion-integrity receipt is ready: `456` candidates audited, `176`
independently verified `learned_full_body_token` candidates, `0` integrity
mismatches, and no non-promotion family claims. Promotion still remains blocked
by unrelated capability/runtime gates, so this is a no-cheat guard improvement,
not a model promotion.
`scripts/maturity_integrity_audit.py` and
`scripts/public_transfer_readiness_refresh_v1.py` now consume the same
promotion-integrity receipt. The maturity audit reports
`promotion_integrity_ready=true`, and public-transfer readiness carries the
same `176` independently verified promotion candidates plus the current
`2227`-record
VIEA receipt before any public-transfer readiness state can be considered
current. This closes a downstream bypass path; the remaining wall is semantic
public transfer, not candidate-family accounting.

- `scripts/trainable_transformer_hybrid_code_generator_v1.py` previously fed
  `category`, `decoder_contract.return_shape`, `decoder_contract.type_family`,
  and `decoder_contract.required_constructs` into `row_to_text()`;
- `prompt_contract_score()` previously gave a direct bonus when
  `task.category == action.action_id`;
- candidate rows from that path previously claimed learned-generation and
  promotion eligibility even though the path selected from a fixed
  hand-authored action renderer.

Those claims are retired. The path is now treated only as a neural
action-selector baseline with a fixed renderer. It cannot support learned code
generation, broad transfer, model growth, or promotion claims unless a separate
blind information-flow audit is GREEN and a genuinely learned generator beats
the baseline under prompt/signature-only inference.

The new candidate-integrity harness recomputes candidate family from provenance,
mode, code shape, and artifact lineage instead of trusting row flags such as
`benchmark_promotion_eligible` or `token_level_code_generation_learned`.
The blind information-flow audit is now a separate required promotion gate:
`scripts/blind_information_flow_audit.py`.

Current evidence:

- VIEA/PlanForge execution spine:
  `scripts/theseus_plan_compiler.py` now emits ASI Stack protocol records for
  every compiled plan node rather than only Theseus-specific route packets. The
  latest report `reports/theseus_plan_compiler.json` is `GREEN` with `8`
  compiled goals, `22` nodes, `22` trace rows, `716` ASI Stack record IDs,
  `64` goal governance record groups, `0` hard gate failures, `0` public
  training rows, `0` external inference calls, and `0` fallback returns.
  Node-level records include semantic atoms/nodes, typed jobs, runtime adapter
  invocations, authority transitions, authority-use receipts, context ABI
  records, context transactions, context adequacy records, resource budgets,
  costed routes, generation mode records, failure boundary maps, artifact graph
  records, evidence transitions, proof-carrying claim envelopes, routing
  decisions, and simulation contracts. The proof/substrate spine profile
  `proof_contract_and_substrate_adoption_spine_v1` is also `GREEN`: the latest
  proof-carrying contract gate emits nine schema-shaped
  `proof_contract_receipt_record` rows and three schema-shaped
  `substrate_adoption_record` rows, which VIEA materializes as canonical
  `proof_contract_receipt` and `substrate_adoption` families. Those records are
  structural/proof-boundary evidence only; they do not claim model quality,
  runtime speed, memory efficiency, context length, public transfer, or learned
  generation. The VIEA materialized view also now normalizes
  `artifact_graph_record`, `context_transaction`, `costed_route`,
  `compressed_artifact_record`, `compression_receipt`, and
  `compact_generative_record` payloads to the
  current AI_book schema-required field sets; `summary.schema_payload_gap_count=0`
  is part of the gate, so skeletal artifact/context/route/compression wrappers
  cannot silently satisfy roadmap evidence. Goal-level records include intent
  contracts, command contracts, reference traces, constitutional predicates,
  agency rights checklists, value conflict records, governance rights, and
  research backlog entries. The bounded private execute path also passed:
  `reports/viea_execution_spine.json` is `GREEN` with `14` compiled private
  cases, `14` leases, `14` checkpoints, `168` runtime ASI Stack execution
  records, `1.0` compiled useful completion rate, `14` metadata-only
  training-evidence rows, `2` verified procedural tools, and no public
  training, runtime external inference, or fallback returns. Runtime records
  now include authority transitions, authority-use receipts, adapter
  invocations, VCM context transactions, context adequacy, resource budgets,
  costed routes, generation-mode boundaries, failure boundaries, artifact
  graphs, evidence transitions, and proof-carrying claims for every executed
  deterministic tool case.
  This `PROJECT_STATE` surface is a Virtual Context Memory consumer: it is a
  human current-state view over VCM `context_packet` outputs, claim/evidence
  reports, and registry-owned support states rather than a separate memory
  lane.
- Assistant/runtime vertical slice:
  `scripts/theseus_assistant_runtime.py` now emits prompt-scoped VIEA trace
  records for assistant calls: intent contract, command contract, PlanForge
  DAG, context ABI, `context_transaction`, `context_adequacy`, typed job,
  runtime adapter invocation, authority transition, authority-use receipt,
  resource budget, generation mode, failure boundary, artifact graph, claim,
  evidence transition, residual, and policy optimization. Nontrivial
  code/tool/planning calls fail their runtime gate if those records are
  missing. `reports/vcm_task_context_bridge.json` is `GREEN` with `9/9` task
  families ready and `7/7` high-priority families ready. The product assistant
  path now also consumes `reports/vcm_context_governor.json`:
  `reports/theseus_assistant_vcm_governor_smoke.json` is `GREEN`, the hard
  `vcm_context_governor_ready` gate passes, adequacy state is
  `governed_sufficient`, mission brief status is `ready`, deletion closure is
  `closed`, and no public training rows, runtime external inference, or
  fallback returns are recorded. The standalone planner now consumes the same
  VCM governor receipt: `reports/theseus_plan_compiler.json` is `GREEN`,
  reports `vcm_context_governor_ready=true`,
  `vcm_context_adequacy_governed_node_count=19`, zero failed gates, and zero
  public training rows, runtime external inference, or fallback returns. The
  private verifier spine smoke also consumes the same governor:
  `reports/private_verifier_spine_smoke.json` is `GREEN`, emits `12` VIEA
  verifier records, includes `context_transaction` and `context_adequacy`, and
  reports `vcm_context_adequacy_state=governed_sufficient_for_verification`.
  `reports/theseus_assistant_e2e.json` is `YELLOW`, not `RED`: `4/4` assistant
  cases pass, deterministic tool evidence is `GREEN` with `15` results and
  `1.0` tool-on solve rate, `6` metadata-only dogfood rows were written, and
  the new VIEA trace gate passes. The remaining warning is the true generator
  wall: the private code probe is safe with `36` eligible candidates, `0`
  integrity mismatches, `0` fallback returns, and `0.0` selected/pass-if-any
  semantic pass.
- Strict generator source contract:
  `scripts/strict_generator_mlx_decode_eval.py` now constructs strict
  generator source text with `prompt_signature_only_v1`: prompt, entry point,
  visible argument-count signature, identifier pieces, and tokenizer subword
  repair only. It no longer accepts arbitrary configured row fields in the
  strict decode/adaptation source builder, and the new
  `project_theseus_strict_generator_source_text_audit_v1` gate hard-fails on
  forbidden field markers or solution/test fragments. The bounded report
  `reports/strict_generator_mlx_decode_eval_prompt_signature_broad8_v1.json`
  is `YELLOW`: source audit clean, `0` public training rows, `0` external
  inference calls, `0` fallback returns, `2` integrity-verified
  transformer/hybrid candidates emitted, syntax/static coherence clean for
  emitted rows, but `0/8` intended-behavior pass. This replaces looser
  prompt/operation-tag diagnostic evidence for strict-generator promotion
  purposes. The current wall remains semantic body generation from prompt and
  signature, not source leakage, syntax, or candidate integrity.
- Strict MLX generator repair update:
  the strict source contract now allows only prompt-derived intent and
  type-shape tags in addition to the prompt/signature fields above; operation
  tags, categories, return-shape labels, tests, expected answers, solution
  bodies, source task ids, and hidden benchmark/card labels remain forbidden.
  `scripts/strict_generator_mlx_private_adaptation.py` now rebuilds rehearsal
  rows through the same strict prompt/signature source builder instead of
  reusing cached legacy `source_text`, and explicit zero loss weights are
  preserved rather than replaced by defaults. The RED report
  `reports/strict_generator_mlx_private_adaptation_prompt_signature_tags_v1.json`
  caught the old rehearsal contamination and must not be used for claims. The
  clean reports
  `reports/strict_generator_mlx_private_adaptation_prompt_signature_tags_rehearsal_clean_v1.json`
  and
  `reports/strict_generator_mlx_private_adaptation_prompt_signature_tags_guard_clean_v2.json`
  are `GREEN` with source audits clean, `0` public training rows, and `0`
  external inference calls. Decode search now expands beam/branch width when
  `--output-top-k` requests a broad candidate set. The best current strict
  clean decode report,
  `reports/strict_generator_mlx_decode_eval_prompt_signature_tags_guard_clean_v2_search_semantic_head_broad8_labelled_v2.json`,
  is still `YELLOW`: `34` integrity-clean transformer/hybrid candidates,
  `34/34` generated candidates with private verifier correctness labels
  attached, `34` runtime-loaded generated attempts, `0` generated intended-
  behavior passes, `0` fallback returns, `0` public training rows, and `0`
  external inference calls. The report has a hard gate requiring private
  verifier correctness labels to attach to generated candidates. It also
  removes stale `visible_operation_tags` provenance from strict MLX rows; the
  recorded generation inputs are now prompt, entry point, callable signature,
  prompt-derived intent/type-shape tags, identifier parts, and visible
  subwords. This moves the wall from leakage, syntax, loadability, and label
  plumbing toward learned semantic/dataflow correctness after loadable
  candidates exist.
  A verifier-label replay split now exists for private train rows:
  `private_train_replay` copies private train rows into a verifier-only replay
  view, excludes configured family-disjoint holdout families, and marks the
  evidence as training-signal replay rather than heldout or promotion evidence.
  The strict-guard smoke
  `reports/strict_generator_mlx_decode_eval_private_train_replay_labelled_smoke_v1.json`
  is `RED` because no admissible candidates survived the strict guard. The
  no-guard replay smoke
  `reports/strict_generator_mlx_decode_eval_private_train_replay_labelled_no_guard_smoke_v1.json`
  is `YELLOW`: it emitted `16` candidates and attached verifier labels to all
  generated rows, with `8` runtime-loaded attempts, `8` lint/parse failures,
  `0` intended-behavior passes, and integrity mismatches on invalid/no-function
  or inert candidates. This is useful negative replay signal for future
  rejection/unlikelihood training; it is not evidence of heldout transfer.
  The first guarded negative-replay objective is now implemented in
  `scripts/strict_generator_mlx_private_adaptation.py`. It consumes only
  failed `private_train_replay` candidates with attached private verifier
  labels, rebuilds prompt/signature source text from the original private row,
  and subtracts a bounded failed-candidate CE term from the supervised private
  objective. The high-weight smoke
  `reports/strict_generator_mlx_private_adaptation_negative_replay_smoke_v1.json`
  is `RED` because heldout LM loss worsened, which is a valid failed setting.
  The lower-weight smoke
  `reports/strict_generator_mlx_private_adaptation_negative_replay_smoke_low_weight_v1.json`
  is `GREEN`: heldout LM loss improved from `2.552511` to `1.715164`, negative
  replay was active on `16` failed private replay rows, `0` public training
  rows, `0` external inference calls, and no fallback/template/router credit.
  Decode from that checkpoint in
  `reports/strict_generator_mlx_decode_eval_negative_replay_smoke_low_weight_private_train_replay_v1.json`
  is `YELLOW`: `64` generated candidates, `48/64` integrity-verified rows,
  syntax pass `0.8125`, runtime load rate `0.708333`, inert-stub rate
  `0.359375`, nontrivial-return rate `0.21875`, and still `0` intended-
  behavior passes. This moves the current wall again: replay labels can now
  train the generator without cheating, but semantic/dataflow correctness is
  still not solved.
  A verifier-stage-weighted replay variant also exists:
  `reports/strict_generator_mlx_private_adaptation_negative_replay_stage_weighted_smoke_v1.json`
  is `GREEN` and applies reward-inverse negative weights (`1.0` for
  lint/parse failures, `0.475` for runtime-loaded wrong answers) without public
  data or runtime teacher use. Its decode report,
  `reports/strict_generator_mlx_decode_eval_negative_replay_stage_weighted_private_train_replay_v1.json`,
  is `YELLOW` but worse than uniform replay on this smoke: integrity-verified
  rows fell from `48/64` to `42/64`, syntax pass from `0.8125` to `0.65625`,
  runtime load from `0.708333` to `0.604167`, nontrivial-return rate from
  `0.21875` to `0.015625`, and intended-behavior pass stayed `0`. Do not
  promote stage-weighting from this evidence; keep it as a negative ablation.
  The decode evaluator now supports private-train replay tiers selected from
  existing private rows rather than new synthetic suites:
  `simple_return`, `loop_accumulate`, and `algorithmic_small`. After configured
  family-disjoint holdout exclusion, the current inventory is `125`
  `simple_return` rows, `1250` `loop_accumulate` rows, and `1625`
  `algorithmic_small` rows. The current uniform negative-replay checkpoint on
  the `simple_return` tier,
  `reports/strict_generator_mlx_decode_eval_negative_replay_low_weight_simple_return_replay_v1.json`,
  is `YELLOW`: `64` candidates, `64/64` integrity-verified, syntax pass `1.0`,
  runtime load `0.833333`, `0` fallback returns, `0` public training rows, and
  `0` intended-behavior passes. The residual is now sharply type/return
  handling: `48` generated rows failed with `type_handling` labels and
  nontrivial-return rate was `0.0`. This is the next correctness wall.
  A targeted simple-return adaptation then trained only on the existing
  private `simple_return` tier with return-expression loss boosted. The
  adaptation report
  `reports/strict_generator_mlx_private_adaptation_simple_return_return_expr_v1.json`
  is `GREEN`: it selected `125` eligible private simple-return rows after
  holdout exclusion, trained on `100`, held out `25`, improved heldout LM loss
  from `2.289181` to `0.095394`, and used `0` public training rows, `0`
  external inference calls, and `0` fallback/template/router credit. The
  follow-up replay decode
  `reports/strict_generator_mlx_decode_eval_simple_return_return_expr_v1.json`
  is still `YELLOW`: `64/64` candidates are independently integrity-verified,
  syntax pass is `1.0`, runtime load remains `0.833333`, and nontrivial-return
  rate moves from `0.0` to `1.0`, but intended-behavior pass remains `0/64`.
  Sample generated bodies now return real expressions such as tuple/list
  indexing, but they return the wrong value. The wall has therefore moved from
  inert/simple-return shape to answer/value semantics and prompt-conditioned
  dataflow.
  A same-source private pairwise replay objective now exists in
  `scripts/strict_generator_mlx_private_adaptation.py`: for failed
  `private_train_replay` candidates it trains the checkpoint to prefer the
  admitted private solution body over the failed generated body under the same
  prompt/signature source, with `0` public rows, `0` external inference, and no
  candidate-generation credit. The first pairwise smoke,
  `reports/strict_generator_mlx_private_adaptation_simple_return_pairwise_value_v1.json`,
  is `GREEN` and selected `64` failed private replay pairs, but its decode
  report
  `reports/strict_generator_mlx_decode_eval_simple_return_pairwise_value_v1.json`
  remains `YELLOW` with `0/64` intended-behavior pass. This is a clean
  negative result: naive same-source accepted/rejected margin pressure did not
  move behavior on this slice.
  Two prompt-visible target-weighting hooks also now exist for this exact wall:
  `default_parameter_return_loss_boost` weights admitted private target lines
  such as `return other` when the prompt/signature source text visibly contains
  default-like intent, and `truthiness_guard_loss_boost` weights guards such as
  `and data` when the prompt visibly mentions empty input. The default-only
  adaptation
  `reports/strict_generator_mlx_private_adaptation_simple_return_default_value_v1.json`
  is `GREEN`, but its decode report
  `reports/strict_generator_mlx_decode_eval_simple_return_default_value_v1.json`
  stayed `0/64` and overproduced complex indexing expressions. Enabling the
  existing top-level-return decode guard on the return-expression checkpoint,
  `reports/strict_generator_mlx_decode_eval_simple_return_return_expr_top_level_guard_v1.json`,
  caused candidates to include `return other` (`16/64`) but still no
  `and data` truthiness guard and no semantic pass. The combined default plus
  truthiness adaptation,
  `reports/strict_generator_mlx_private_adaptation_simple_return_default_truthiness_v1.json`,
  is `GREEN` and weighted both `return other` and `and data` on `100` private
  rows, but the guarded decode
  `reports/strict_generator_mlx_decode_eval_simple_return_default_truthiness_top_level_guard_v1.json`
  remains `YELLOW` with `0/64` intended-behavior pass. The current narrow wall
  is therefore condition construction and beam/search policy for prompt-visible
  empty/default behavior, not syntax, loadability, missing top-level return,
  public data, teacher use, or fallback/template credit.
  That condition-construction wall now has a bounded constrained-control
  result. `scripts/strict_generator_mlx_decode_eval.py` derives a
  source-condition expectation only from prompt/signature source text and can
  prefer branch construction for visible empty/default sequence contracts. The
  intermediate polarity report
  `reports/strict_generator_mlx_decode_eval_simple_return_source_condition_polarity_prefer_v1.json`
  is retained as a negative artifact because its old adequacy check accepted
  tuple-only and overlong boolean-return bodies even though behavior remained
  `0/64`. The stricter ordered report
  `reports/strict_generator_mlx_decode_eval_simple_return_source_condition_sequence_ordered_prefer_v1.json`
  failed closed with `0` emitted candidates until mandatory indentation was
  restored ahead of source-condition preferences and the verifier-label gate
  was aligned to evaluated traces rather than unexecuted fanout beams. The
  current bounded report
  `reports/strict_generator_mlx_decode_eval_simple_return_source_condition_sequence_indent_strict_v1.json`
  is `GREEN` on `16` existing private simple-return replay rows: `32`
  integrity-verified transformer/hybrid candidates, `16/16` rank-1 and
  pass-if-any intended behavior, source text audit clean, `0` public training
  rows, `0` runtime external inference, `0` fallback returns, and
  `candidate_generation_credit=0` on the source-condition constraint itself.
  This is deliberately not a learned free-generation promotion claim. It proves
  the prompt-visible branch/value contract can be expressed safely through the
  direct-token decoder when constrained; the next learned-generator target is
  to make the model probabilities/internal planning produce the same
  `if isinstance(data, (list, tuple)) and data: return data[0]; return other`
  shape without deterministic source-condition assistance, then test it on
  broader held-out private families before public calibration.
  The first attempt to internalize that constrained target is also complete:
  `reports/strict_generator_mlx_private_adaptation_simple_return_source_condition_internalize_v1.json`
  is `GREEN` and boosted `2,600` prompt-visible source-condition target-token
  positions across `100` admitted private simple-return rows, with clean source
  audits, heldout LM improvement, `0` public training rows, `0` external
  inference, and no candidate-generation credit. The decisive ablation,
  `reports/strict_generator_mlx_decode_eval_simple_return_source_condition_internalize_no_assist_v1.json`,
  is still `YELLOW` with `0/16` intended-behavior pass when source-condition
  assistance is off. It emits more sequence/indexing shape
  (`isinstance(data, (list, tuple))` appears in `48/64` candidates), but still
  emits `0` `and data` guards and no adequate source-condition candidates.
  The next plan-auxiliary pass narrowed that wall further:
  `reports/strict_generator_mlx_private_adaptation_simple_return_source_condition_plan_aux_v1.json`
  is `GREEN` and moves the learned semantic-plan head from `0.0` to `1.0`
  heldout plan accuracy on the same private tier, with all train/eval targets
  in the `SLOT:PLAN_SAFE_HEAD_DEFAULT` family and no public data, external
  inference, or candidate-generation credit. The no-assist decode
  `reports/strict_generator_mlx_decode_eval_simple_return_source_condition_plan_aux_no_assist_planmeta_v1.json`
  proves the distinction cleanly: `64/64` candidates emit a complete
  `SLOT:PLAN_SAFE_HEAD_DEFAULT` prefix, but behavior is still `0/16`; all
  `64` verifier labels fail, source-condition adequacy remains `0/64`, and
  the body emits `0` `and data` truthiness guards. That is the current honest
  wall: the constrained control can solve this private slice and the learned
  coarse plan is now correct, but the body-token decoder still has not learned
  operand choice, truthiness guard placement, or the default-return expression
  without deterministic source-condition assistance.
  A follow-up exact learned decision-prefix mode is now implemented in the
  same strict generator target contract. It trains the decoder to emit
  model-generated slots such as `SLOT:COND_TRUTHY_ARG_DATA`,
  `SLOT:RETURN_GUARDED_HEAD_ARG_DATA`, and `SLOT:RETURN_DEFAULT_ARG_OTHER`
  before `SLOT:BODY_START`; those slots are stripped before Python compilation
  and do not render code or grant learned-generation credit. The fresh private
  smoke checkpoint
  `reports/strict_generator_mlx_pretraining_probe_plan_semantic_slots_body_exact_smoke_v1.json`
  plus adaptation
  `reports/strict_generator_mlx_private_adaptation_simple_return_plan_semantic_slots_body_exact_decision_v1.json`
  cleanly learns the exact decision-prefix targets. The strict no-assist smoke,
  `reports/strict_generator_mlx_decode_eval_simple_return_plan_semantic_slots_body_exact_decision_no_assist_smoke_v1.json`,
  still has `0/4` behavior, proving body tokens alone are not consuming the
  slots reliably. The opt-in prefix-guided diagnostic
  `reports/strict_generator_mlx_decode_eval_simple_return_plan_semantic_slots_body_exact_decision_prefix_guided_ranked_v2.json`
  reaches `16/16` rank-1 and pass-if-any on private simple-return replay after
  adding a prefix-aware ranker: no public data, no external inference, no
  fallback returns, and `candidate_generation_credit=0` for the prefix/ranker
  constraint. That is real progress for the practical transformer/hybrid lane,
  but the claim boundary is explicit: this is learned decision-prefix plus
  constrained/ranked decoding evidence, not unconstrained free-form body-token
  generation and not broad public transfer.
  Multi-tier follow-through now shows where that progress stops. The private
  adaptation path supports deterministic tier-balanced sampling, and the
  strict decoder now prevents incomplete exact-prefix body starts, invalid
  loop targets, and mismatched closing brackets while reporting max-target
  overrides. The balanced checkpoint
  `reports/strict_generator_mlx_private_adaptation_multi_tier_balanced_plan_semantic_slots_body_exact_decision_v1.json`
  is `GREEN`, and the tightened simple-return replay
  `reports/strict_generator_mlx_decode_eval_simple_return_multi_tier_balanced_plan_semantic_slots_body_exact_decision_prefix_guided_cap80_prefixrequired_v1.json`
  is `GREEN` with `16/16` rank-1/pass-if-any, syntax `1.0`, no public data,
  no external inference, and no fallback returns. The same route is still not
  enough for loop/action semantics: even the loop-focused source-contrastive
  private adaptation on `900` existing loop rows,
  `reports/strict_generator_mlx_private_adaptation_loop_focused_source_contrast_plan_semantic_slots_body_exact_decision_v1.json`,
  decodes to `0/8` behavior in
  `reports/strict_generator_mlx_decode_eval_loop_accumulate_loop_focused_source_contrast_plan_semantic_slots_body_exact_decision_cap160_v1.json`.
  Pairwise private replay now composes with semantic-plan, source-contrastive,
  and negative-replay losses instead of silently disabling itself in that
  combined setting. The composite adaptation
  `reports/strict_generator_mlx_private_adaptation_loop_focused_composite_pairwise_plan_semantic_slots_body_exact_decision_v1.json`
  is `GREEN` and proves pairwise/source/plan/negative losses are all active
  under private-only no-credit accounting, but its decode report is RED with
  `0` accepted candidates. The lighter pairwise-only composition
  `reports/strict_generator_mlx_private_adaptation_loop_focused_pairwise_only_plan_semantic_slots_body_exact_decision_v1.json`
  is also `GREEN`, but
  `reports/strict_generator_mlx_decode_eval_loop_accumulate_loop_focused_pairwise_only_plan_semantic_slots_body_exact_decision_cap160_v1.json`
  remains `0/8` with only one valid wrong candidate. The simple-return
  regression
  `reports/strict_generator_mlx_decode_eval_simple_return_multi_tier_balanced_plan_semantic_slots_body_exact_decision_prefix_guided_cap80_regression_v2.json`
  stays `GREEN` at `8/8`, so the narrow working slice was preserved.
  Current wall: learned loop/action body semantics and decode speed, not
  simple-return proof, source leakage, public benchmark permission, or
  fallback/template plumbing.
- ATT-D/source-shape cleanup:
  The former hard cap violation on
  `scripts/neural_seed_token_decoder_comparator.py` is cleared. Visible
  prompt/signature helpers moved to
  `scripts/neural_seed_visible_source.py`, and task-blind static-coherence
  checks moved to `scripts/neural_seed_static_coherence.py`; the original
  comparator re-exports the same names for compatibility. Current
  static decode guards are extracted into
  `scripts/neural_seed_decode_static_guard.py`; shared candidate schema,
  no-cheat evidence, grammar/static/body summaries, and token-decoder gate
  predicates are extracted into
  `scripts/neural_seed_candidate_evidence_summary.py`; expression-value hygiene
  helpers are extracted into `scripts/neural_seed_expression_value_guard.py`;
  governed teacher/self
  code-LM row admission helpers are extracted into
  `scripts/neural_seed_teacher_distillation_rows.py`; semantic route-memory
  helpers for learned plan prototypes, contract fingerprints, contract
  features, and visible-text prototypes are extracted into
  `scripts/neural_seed_route_memory.py`; private target-body guard weighting
  and existing-row tier selection for strict MLX adaptation are
  extracted into `scripts/strict_generator_mlx_adaptation_selection.py`;
  private replay and family/broad heldout row selection are extracted into
  `scripts/strict_generator_mlx_replay_selection.py`; prompt/signature-only
  strict MLX source-text construction and leakage audits are extracted into
  `scripts/strict_generator_mlx_source_text.py`; strict MLX decode row
  stamping, inline integrity summaries, gate construction, checkpoint path
  resolution, and report IO are extracted into
  `scripts/strict_generator_mlx_decode_reporting.py`; specialist-head
  route/profile records for private replay checkpoints are extracted into
  `scripts/strict_generator_mlx_specialist_routing.py`; task-blind strict
  body-token decode hygiene and builtin/method call guards are extracted into
  `scripts/strict_generator_mlx_decode_guards.py`; local corpus admission,
  full-state pretraining row construction, quality filtering, and vocab
  extension summaries are extracted into
  `scripts/neural_seed_full_state_pretraining.py`; matched token decoder
  backends, checkpoint/vocab initialization, supervised token training,
  grammar auxiliary loss, semantic token weighting, and parameter-update
  summaries are extracted into `scripts/neural_seed_token_model_backend.py`;
  task-blind token beam expansion/sorting is extracted into
  `scripts/neural_seed_candidate_generation.py`; report/latest-view IO and
  markdown rendering are extracted into `scripts/neural_seed_report_io.py`.
  Prompt-visible source-condition expectations, learned prefix adequacy,
  loop-plan adequacy, expression closure, and body action-trace summaries are
  extracted into `scripts/strict_generator_mlx_decode_plans.py`. The comparator
  is now `3,196` lines and the strict MLX decoder is now `2,510` lines, both
  below the `3,200` system-efficiency hard limit. Private strict-generator
  target weighting and AST span extraction are extracted into
  `scripts/strict_generator_mlx_adaptation_weights.py`, moving
  `scripts/strict_generator_mlx_private_adaptation.py` down to `3,051` lines.
  Token-decoder rendering, strict-action rendering, semantic-slot rendering,
  deterministic semantic bodies, and body-token decoding are extracted into
  `scripts/neural_seed_token_decoder_rendering.py`, moving
  `scripts/neural_seed_token_decoder_support.py` down to `2,832` lines. There
  are no remaining RED source-size blockers. The strict MLX generator path is
  bound to the existing
  `neural_seed_and_decoder` registry surface and is classified by ATT-D as
  `neural_seed`, not an unowned sidecar. `python3 -m py_compile` passed for the
  comparator, extracted modules, and direct strict-generator consumers.
  `scripts/attd_analyzer.py` is `YELLOW` with hard caps passed and no
  violations; `reports/system_efficiency_audit.json` is now `YELLOW` with `0`
  hard maintainability hotspots.
  The refactored strict MLX decode path also has an executed private smoke:
  `reports/strict_generator_mlx_decode_eval_refactor_smoke_v1.json` is
  `YELLOW` on one family-disjoint private row. It loads the MLX checkpoint,
  emits a candidate, preserves the hard no-cheat gates (`0` external inference,
  `0` public training rows, no fallback returns), and correctly records the
  candidate as integrity-failing/inert instead of counting it as learned
  generation.
- No-cheat audit coverage now scans the extracted neural-seed helper modules
  directly: `scripts/neural_seed_visible_source.py`,
  `scripts/neural_seed_static_coherence.py`,
  `scripts/neural_seed_decode_static_guard.py`,
  `scripts/neural_seed_expression_value_guard.py`,
  `scripts/neural_seed_candidate_evidence_summary.py`, and
  `scripts/neural_seed_teacher_distillation_rows.py`,
  `scripts/neural_seed_full_state_pretraining.py`,
  `scripts/neural_seed_token_model_backend.py`,
  `scripts/neural_seed_candidate_generation.py`,
  `scripts/neural_seed_report_io.py`,
  `scripts/neural_seed_route_memory.py`, plus the original
  comparator/support/generator entrypoints,
  `scripts/neural_seed_token_decoder_rendering.py`, strict MLX
  decode/adaptation/pretraining entrypoints,
  `scripts/strict_generator_mlx_decode_guards.py`,
  `scripts/strict_generator_mlx_decode_plans.py`,
  `scripts/strict_generator_mlx_source_text.py`,
  `scripts/strict_generator_mlx_decode_reporting.py`, and
  `scripts/strict_generator_mlx_specialist_routing.py`. The expanded
  `scripts/blind_information_flow_audit.py` run remains `GREEN` with
  `source_file_count=28` and `static_information_flow_violation_count=0`.
  `scripts/attd_analyzer.py` remains `YELLOW` with hard caps passed and no
  violations, and `scripts/theseus_project_registry.py --gate` remains
  `GREEN`. The remaining control-plane maintenance target is rolling-residue
  cleanup, not this hard-cap blocker.
- License-clean corpus scaling rung:
  `docs/CORPUS_INGRESS_POLICY.md` now records the needed policy separation:
  open/base/pretrained model weights remain forbidden, while license-clean
  text/code corpus ingestion is allowed when manifest-governed, provenance-
  tagged, license-recorded, hash-audited, and decontaminated. The current
  corpus spine report `reports/narrow_corpus_pretraining_spine.json` is
  `GREEN`. It admits `484` local Apple Command Line Tools Python standard-
  library files under `PSF-2.0`, with `1,404,663` rough corpus tokens,
  `1,000,000` encoded tokens, `0.846046` algorithmic-Python token fraction,
  `0` public benchmark payloads admitted, `0` eval-overlap admissions, `0`
  public training rows, `0` external inference calls, and no open/pretrained
  model weights. The tokenizer is from scratch with `7,228` vocab entries and
  `8,088` merges. The matched random-init pretraining checkpoints are
  `checkpoints/narrow_pretraining/symliquid_style_rung_1m_probe.pt` and
  `checkpoints/narrow_pretraining/transformer_control_rung_1m_probe.pt`.
- Strict comparator after the 1M corpus rung:
  `reports/neural_seed_token_decoder_comparator_rung_1m.json` is still `RED`.
  The 1M corpus checkpoints and 512 admitted stdlib function-body warmup rows
  improved syntax but not semantic transfer: in-family no-cheat verifier pass
  stayed `5/24 = 0.208333`, and strict family-disjoint stayed `0/24`.
  The first 1M pass exposed a real decoder bottleneck: the old strict target
  vocabulary had only `179` body tokens, leaving full-state warmup with
  `0.278834` target `<unk>` rate.
- Strict comparator with corpus-extended target vocabulary:
  `reports/neural_seed_token_decoder_comparator_rung_1m_vocab_extended.json`
  is also `RED`, but it resolves the immediate target-vocabulary bottleneck.
  The decoder target vocab is now `1,024`, built from admitted corpus function
  bodies plus private train bodies, with `0` eval/public leakage. Full-state
  warmup target `<unk>` rate fell to `0.059775`. SymLiquid in-family STS-on
  syntax improved to `0.916667`; SymLiquid family-disjoint syntax stayed
  `1.0`; transformer family-disjoint syntax improved to `0.666667`. Semantic
  behavior did not move: in-family remains `5/24`, family-disjoint remains
  `0/24`, and both SymLiquid and transformer remain tied on strict verifier
  pass. Candidate integrity for this run is `GREEN` in
  `reports/candidate_integrity_rung_1m_vocab_extended.json` with `0`
  mismatches, `357` learned full-body token rows, `255` integrity-verified
  learned candidates, and `102` learned syntax-invalid candidates. Blind
  information-flow is `GREEN` in
  `reports/blind_information_flow_audit_rung_1m_vocab_extended.json` with `0`
  invalid claims or information-flow violations.
- Current interpretation:
  the previous 80k/20-example canary was genuinely starved. The new 1M rung
  and target-vocab extension prove the pipeline can admit real license-clean
  code and improve syntactic support without cheating. They do not prove broad
  semantic transfer. The 10M rung below is the decisive follow-up for the
  active corpus-scaling goal.
- 10M corpus scaling result:
  `reports/narrow_corpus_pretraining_spine_10m.json` is `GREEN`. It admits
  `3,219` license-clean Python files from explicit PSF/BSD/MIT/Apache local
  sources, with `12,081,298` rough tokens, `10,000,000` encoded tokens,
  `0.814123` algorithmic-Python token fraction, `0` public benchmark payload
  admissions, `0` eval-overlap admissions, `0` public training rows, `0`
  external inference calls, and no open/pretrained model weights. The 10M
  tokenizer has `10,191` vocabulary entries and `11,896` merges. Matched
  random-init checkpoints are in `checkpoints/narrow_pretraining_10m/`.
- 10M strict comparator result:
  `reports/neural_seed_token_decoder_comparator_rung_10m_vocab_extended.json`
  is `RED`. It uses the 10M tokenizer/checkpoints, `768` admitted function-body
  warmup rows, a `1,024` strict decoder target vocabulary extended from
  admitted corpus bodies, and unchanged private in-family/family-disjoint eval
  tasks. In-family no-cheat verifier pass remains `5/24 = 0.208333`; strict
  family-disjoint remains `0/24`. SymLiquid in-family syntax is `0.777778`;
  transformer in-family syntax is `0.569444`; SymLiquid family-disjoint syntax
  is `1.000000`; transformer family-disjoint syntax is `0.486111`.
  Candidate integrity is `GREEN` in
  `reports/candidate_integrity_rung_10m_vocab_extended.json` with `0`
  mismatches and `245` integrity-verified learned-token candidates. Blind
  information-flow is `GREEN` in
  `reports/blind_information_flow_audit_rung_10m_vocab_extended.json` with `0`
  invalid claims.
- Scaling-curve verdict:
  `reports/corpus_scaling_ladder_curve_v1.json` is the current curve report.
  The largest completed rung is `10,000,000` encoded tokens, exactly `125.0x`
  the old 80k canary reference. Family-disjoint stayed `0/24` across the 1M
  and 10M code-rich rungs while audits stayed green. That rules out simple
  corpus absence as the immediate explanation at this small-model/direct-body-
  token decoder budget. The next wall is architecture/objective/decoder
  semantic planning: the model can increasingly emit syntactically plausible
  bodies, but it is not choosing the right held-out algorithms or return
  behavior. Neither SymLiquid nor transformer earns promotion; transformer has
  better LM eval loss, but both tie on strict verifier behavior.

- Superseded foundation canary:
  the earlier 80k-token project-internal canary is superseded by the 1M
  PSF-licensed code rung above. Its value was proving the pretraining plumbing,
  not behavior. Do not use the older 40-document/1,969-vocab checkpoint as the
  current corpus scale claim.
- Blind information-flow audit:
  `reports/blind_information_flow_audit_rung_1m_vocab_extended.json` is
  `GREEN` for the current comparator path. It audits the strict token-decoder
  comparator/support sources, the token-decoder comparator config, the
  rung-1M-vocab-extended candidate manifest, and the rung-1M-vocab-extended
  strict report. Invalid claim count is `0`.
- Strict prompt/signature body-token comparator:
  `reports/neural_seed_token_decoder_comparator_rung_1m_vocab_extended.json` is
  `RED`, and that is the honest current wall. The run used natural-prompt
  private rows, prompt plus entry-point inference only, direct body-token
  targets, no semantic-slot renderer, no contract semantic beam, no
  structural-action adapter, no fallback terminal return, no public data, and
  matched parameter budgets. The latest completed report is the guarded
  two-row teacher-training run after utility quarantine plus the 1M
  license-clean corpus checkpoints and corpus-extended target vocabulary: `2`
  admitted executable code-LM rows consumed, `1` verifier-accepted teacher row
  quarantined for negative strict utility (`private_state_window_relay_v1`),
  `0` held-out-family teacher rows, `0` public training rows, and runtime
  external serving forbidden.
- Direct body-token result:
  the strict comparator emitted `456` candidate rows, including `360`
  `learned_full_body_token` rows and `96` quarantined fallback/baseline rows.
  No-cheat in-family verifier pass remains `5/24 = 0.208333`; strict
  family-disjoint verifier pass remains `0/24`; family overlap, prompt overlap,
  entry-point overlap, task-id overlap, solution hash overlap, and target-token
  template overlap are all `0`. The latest 1M-corpus plus target-vocab-
  extension run improved syntax and reduced corpus warmup `<unk>` pressure, but
  not behavior. In-family STS-on syntax is `0.916667` for SymLiquid-style and
  `0.500000` for transformer-control; strict family-disjoint STS-on syntax is
  `1.000000` for SymLiquid-style and `0.666667` for transformer-control.
  Fresh candidate integrity verifies `255` learned-token candidates and flags
  `102` learned-token syntax-invalid candidates. Semantic verifier pass is
  still `0/24` on the strict
  family-disjoint split. Fallback returns remain `0`.
  SymLiquid and transformer are tied on current strict verifier behavior; neither
  can support promotion or broad-transfer claims. The smallest next private
  repair is no longer basic syntax for the SymLiquid/in-family path. The
  family-disjoint residuals still show semantic body failure despite the
  foundation canary: fragment mixing, variable-lifetime collapse, wrong local
  roles, malformed transformer held-out branches, and wrong return behavior.
  Examples include
  `bpg_normalize_filter_sort` splicing unrelated expression fragments into
  filter/sort bodies, `bpg_stable_dedup` and `bpg_top_k_frequent` using
  undefined or wrong-shape locals, `bpg_stdin_pair_sums` mixing graph/component
  fragments with stdin parsing, and `bpg_group_records`/`bpg_threshold_labels`
  collapsing to partial try/except bodies. The next private repair should target
  the pretraining ladder itself: expand admitted corpus size and tokenizer
  budget, train longer from random init, and reuse the same strict
  family-disjoint split before returning to teacher distillation or public
  calibration. If larger clean pretraining still leaves family-disjoint at
  `0/24`, the next wall is architecture/objective rather than corpus absence.
- Strict candidate integrity:
  `reports/candidate_integrity_rung_1m_vocab_extended.json` is `GREEN` as an
  integrity/classification audit, not as a behavior win. It recomputes `360`
  learned/fallback candidate attempts with `0` mismatches. `255`
  learned-token candidates are syntax-valid, nontrivial integrity candidates;
  `102` learned-token candidates remain
  syntax-invalid.
- Registry/governance:
  `reports/theseus_project_registry.json` is `GREEN` under the
  abstraction/implementation governance kernel plus the Stable Capability
  Field semantic-ABI layer. The SCF contract now tracks the public-release
  1.0 paper: exact content identity, source-event evidence, scoped
  qualification claims, caller-bound route validation, expiring fail-closed
  leases, sealed adaptation epochs, migration solvency, evaluator overlap, and
  lifecycle/governance controls. Abstraction gaps are `0`, stable capability
  field gaps are `0`, stable capability field red/yellow health counts are
  `0/0`, implementation routing blockers are `0`, routing-eligible
  implementations are `11`, stale declared report outputs are `0`, and hard
  registry governance violations are `0`. The Octopus router is now
  registry-gated and SCF-gated.
- Control-plane freshness:
  `reports/theseus_control_plane.json` now has `0` stale reports and `0`
  missing reports. The remaining `RED` state is no longer stale evidence; it is
  the real wall: public transfer, learned-generator semantic quality, promotion
  coherence, iteration speed, and assembly debt.
- Neural action-selector baseline:
  `scripts/trainable_transformer_hybrid_code_generator_v1.py` is only a
  neural action-selector over a fixed renderer. It is useful as a tool/router
  baseline, but it is not learned code generation and cannot support
  learned-generation promotion or broad-transfer claims.
- Stale/adapter evidence:
  semantic-slot renderers, structural adapters, private ngrams, body-template
  selectors, and previous clean64 `64/64` reports are diagnostic only. They
  may inform residual mining, but they cannot count as promotion-grade learned
  generation.

## Candidate Family Ablations

Private replay now supports real family-filtered ablations. Family filters no
longer pass through the old promotion gate first.

- Structural adapter, current closure fanout:
  `reports/private_candidate_replay_contract_audit_reality_harness_structural_adapter.json`
  is `YELLOW` under the functional-promotion gate. `16/16` structural
  candidates compile and load, but selected intended-behavior pass is `0/16`
  and functional promotion is `0/16`.
- Learned full-body token, fresh token-generator smoke:
  `reports/private_candidate_replay_contract_audit_reality_harness_token_learned_full_body.json`
  is `YELLOW`. On the first `24` private residual-smoke tasks, selected pass is
  `1/24 = 0.041667`; selected functional promotion is `1/24`; all-candidate
  functional promotion is `2/36`; selected compile/load/lint is `7/24`.
- Private ngram body, fresh token-generator smoke:
  `reports/private_candidate_replay_contract_audit_reality_harness_token_private_ngram.json`
  is `YELLOW`. Selected pass is `0.0`; private ngrams remain diagnostic/private
  pressure only.
- Canonical transformer/hybrid/action-selector clean-split replay:
  the previous `64/64` reports are invalid for learned-generation and
  promotion claims. They may be used only as evidence that a fixed action
  catalog can solve tasks when answer-family metadata leaks into ranking. They
  must be regenerated under `scripts/blind_information_flow_audit.py` before
  any score is treated as an action-selector baseline.
- Combined learned plus private ngram replay does not improve selected
  functional promotion over learned-only replay in the older token-generator
  smoke: both are `1/24`.

`reports/private_heldout_transfer_baseline_v1.json` is `YELLOW` because the
default post-v4 private train/eval pair has exact prompt and code overlap on
the heldout rows. That replay score is diagnostic only.

`reports/private_heldout_transfer_baseline_v1_disjoint.json` is the current
valid private held-out transfer baseline. It derives a clean split from
`data/private_code_curriculum/code_lm_closure_private_residual_repair_v3_private_proof.jsonl`
with `2560` train rows and `64` heldout rows. Exact prompt/code/template
overlap is `0/0/0`; max prompt/code/template ngram Jaccard is
`0.000819/0.004735/0.002477`; the replay report matches the heldout manifest;
public training rows are `0`; external inference calls are `0`.

`reports/private_heldout_transfer_baseline_v1_canonical_transformer_hybrid_clean64_v1.json`
is invalid for learned-generation or promotion claims. Exact prompt/code
overlap was not enough; blind information-flow also has to prove the generator
and ranker did not receive answer-family metadata.

The older verified `learned_full_body_token` clean result remains bad:
selected functional promotion is `0/64`, pass-if-any is `0/64`, and
all-candidate functional promotion is `0/16` in
`reports/private_candidate_replay_contract_audit_canonical_ablation_old_learned_template_like_clean64_v1.json`.
That should now be treated as the old token-decoder wall, not the practical
survival lane wall. Structural adapters and private ngrams still must not be
counted as promotion-grade learned generation.

No fallback returns, constant returns, public training rows, or external
inference calls were credited in these audits.

## Architecture Verdict

There is no current promotion-grade learned code-generation lane. The canonical
practical lane is now the strict prompt/signature direct body-token comparator,
and it currently fails on semantic transfer: `0/24` selected verifier pass on
the family-disjoint split, with `144/144` generated held-out-family rows
eligible under the no-cheat filter. Syntax is no longer the only wall. The real
blocker is selecting and emitting the right algorithm, return shape, IO
contract, and variable flow from prompt/signature-only evidence.

Treat the fixed-renderer neural action selector as a baseline/tool router only.
Treat private ngrams, semantic-slot renderers, structural adapters, and old
body-template paths as diagnostics or baselines, not learned generation.

SymLiquid remains protected as a discovery lane, but it is not currently proven
to drive code-generation wins. The transformer arm is the practical survival
control, but it also currently fails on strict body-token generation. Do not
claim pure free-form generation, transformer/hybrid survival-lane superiority,
or SymLiquid code-generation superiority from invalid clean64 or renderer-based
reports.

Older SymLiquid/transformer comparator manifests are not current promotion
manifests. Reality-harness probes against those artifacts produced no eligible
family-filtered replay candidates under this harness, so they remain comparator
diagnostics rather than promotion-grade code-generation evidence.

## Evidence Retention

`scripts/theseus_artifact_retention.py` now recognizes current runtime compact
STS checkpoint families. The older
`reports/student_code_lm_checkpoint_runtime_compact_sts_baseline_guard_private_v1.json`
checkpoint was archived through the manifest-backed retention path, reclaiming
about `0.19 GiB` while leaving a resolver-readable pointer at the original path.
The live source-bound checkpoint used by current fanout was not archived.

## Public Calibration Policy

Public benchmarks remain calibration only. Do not train on public benchmark
prompts, tests, hidden tests, solutions, traces, or answer templates.

Run the next broad public calibration only after a fresh current-source
candidate manifest has:

- `reports/blind_information_flow_audit.json` at `GREEN`;
- `0` candidate-integrity mismatches;
- verified learned-generation candidates on the target path, or a clearly
  labeled tool/router baseline that is reported separately and cannot promote;
- family ablations showing nontrivial selected/pass-if-any semantic quality
  under prompt/signature-only inference;
- promotion reports that break scores down by recomputed candidate family.

The next broad public run should be treated as measurement, not training data.

## Next Repair Target

Repair direct body-token syntax and semantic quality before another capability
claim. The next barrier is not public benchmark permission, a budget registry,
or loadability on the contaminated clean private action-catalog split. It is
that the real prompt/signature-only learned generator now emits more loadable
Python but still passes `0/24` strict family-disjoint private tasks.

Primary subtargets:

- keep `scripts/blind_information_flow_audit.py` green before candidate
  promotion, public calibration, or model growth can proceed;
- build or select a real learned generator that synthesizes candidate bodies
  without fixed action-catalog lookup or semantic-slot rendering;
- improve semantic verifier pass rate under the existing prompt/signature-only
  comparator by training/ablating algorithm planning, return-shape realization,
  IO-contract handling, variable lifetime, and statement/block grammar without
  answer-family metadata;
- compare SymLiquid and transformer arms under matched compute only after both
  emit syntactically valid direct body-token candidates;
- keep public benchmarks as measurement only and never train on their prompts,
  tests, hidden tests, solutions, traces, or answer templates;
- keep SymLiquid as a protected matched-compute comparator, but do not let it
  block the practical assistant path unless it wins repeated current-source
  functional replay evidence.

### 2026-06-25 Loop-Body Decode Update

The loop/action strict generator path is now past the previous zero-row
admissibility failure, but it is still not semantically useful enough to
promote.

Evidence:

- `reports/strict_generator_mlx_decode_eval_simple_return_multi_tier_balanced_plan_semantic_slots_body_exact_decision_prefix_guided_cap80_action_trace_replay_regression_v1.json`
  is `GREEN`: simple-return replay remains `8/8` rank-1 and pass-if-any with
  `1.0` syntax and accepted-candidate rates. The loop-specific search ordering
  no longer hijacks visible safe-head/default-return contracts.
- `reports/strict_generator_mlx_private_adaptation_loop_operation_from_balanced_v1.json`
  is `GREEN`: private loop-operation weighting matched `256/256` admitted
  private loop rows, improved heldout LM loss from `1.160228` to `0.780920`,
  and used `0` public rows, `0` runtime/external inference, and `0`
  fallback/template/router credit.
- `reports/strict_generator_mlx_decode_eval_loop_operation_from_balanced_cap8_v1.json`
  is `YELLOW`: with the older forced-update scaffold it emits `15` generated
  loop candidates with `1.0` syntax and better runtime-load rate, but verifier
  pass remains `0.0` and all candidates are still tagged
  `shallow_identity_accumulation`.
- `reports/strict_generator_mlx_decode_eval_loop_operation_from_balanced_cap8_no_forced_update_v3.json`
  is `YELLOW`: after removing the hand-coded `acc.append(loop_var)` update
  injection, the current learned path emits only `2` loadable candidates on the
  cap-8 loop slice, still with `0.0` verifier pass; failures shift to
  `early_return_inside_loop` plus residual shallow identity behavior.
- `reports/strict_generator_mlx_decode_eval_loop_operation_from_balanced_cap8_action_trace_v1.json`
  attaches `body_action_trace` to generated candidates. The cap-8 action trace
  explicitly reports `loop_without_decision_or_state_update`,
  `shallow_identity_accumulation`, `early_return_inside_loop`,
  `missing_numeric_accumulation`, `missing_list_construction`, and
  `missing_windowed_finalizer` on the surviving loop candidates.
- `reports/strict_generator_mlx_private_adaptation_action_trace_replay_v1.json`
  is `GREEN`: action-trace-aware same-source private replay consumed the failed
  private candidates above, boosted `66` accepted-body token positions and `8`
  failed-candidate token positions from their `body_action_trace` mismatch
  labels, and improved heldout LM loss from `0.769207` to `0.669650`.
- `reports/strict_generator_mlx_decode_eval_action_trace_replay_cap8_v1.json`
  is still `YELLOW`: the action-trace replay checkpoint emits `2` loadable
  loop candidates with `1.0` syntax, but verifier pass remains `0.0` and the
  residual labels remain early loop returns plus shallow identity behavior.
- `reports/strict_generator_mlx_decode_eval_action_trace_replay_cap8_summary_probe_v1.json`
  keeps that result `YELLOW` and now carries a report-level
  `body_action_trace` summary. The summary counts `2/2` traced loop candidates,
  `2` early returns inside loops, `3` shallow identity accumulations, and
  mismatch labels for missing decision/state update, list construction,
  numeric accumulation, and windowed finalization. This makes the failure
  machine-readable without treating the trace as candidate-generation credit.
- `reports/strict_generator_mlx_decode_eval_action_trace_replay_cap8_loop_return_blocked_v1.json`
  is `RED`: after preventing loop-plan exploration from proposing a new
  `return` line while still inside the loop, the same checkpoint emits `0`
  loop candidates. This is the sharpest current diagnosis: the learned body
  decoder is relying on an invalid nested-return escape and does not yet select
  valid update/finalizer actions under the prompt/signature-only contract.
- `reports/strict_generator_mlx_private_adaptation_loop_statement_action_v1.json`
  is `GREEN`: the new exact loop-statement span weighting matched `192/192`
  admitted private loop rows and boosted `7,624` target-token positions across
  loop-body decisions, loop-body updates, and top-level finalizers while still
  using `0` public rows, `0` external inference, and `0` candidate-generation
  credit. Heldout LM loss improved from `0.669650` to `0.619043`.
- `reports/strict_generator_mlx_decode_eval_loop_statement_action_trace_label_cap8_v1.json`
  is `YELLOW`: the span-weighted checkpoint now emits `2` loop candidates
  without `early_return_inside_loop`; the current failure shifts to
  `continue` before the update, reported as
  `unreachable_loop_update_after_control_flow`, plus residual shallow identity,
  missing numeric accumulation, missing list construction, and missing windowed
  finalization. Verifier pass remains `0`.
- `reports/strict_generator_mlx_private_adaptation_loop_statement_action_unreachable_v1.json`
  is `YELLOW`: replay from the new unreachable-update candidates consumed the
  `unreachable_loop_update_after_control_flow` label and improved heldout LM
  loss from `0.619043` to `0.568809`, but the follow-up decode
  `reports/strict_generator_mlx_decode_eval_loop_statement_action_unreachable_cap8_v1.json`
  is `RED` with `0` loop candidates. This is negative evidence that stronger
  control-flow penalty removes the bad candidates before the model has learned
  a replacement update/finalizer path.
- `reports/strict_generator_mlx_private_adaptation_loop_statement_reachable_v1.json`
  is `GREEN`: the refined span objective keeps decision spans as context but
  excludes bare positive control-flow terminal tokens from update/finalizer
  weighting. `NAME:continue` is no longer a positively boosted target token;
  `103` continue positions and `15` loop-return positions are explicitly
  reported as excluded positive tokens. Heldout LM loss still improves
  `0.669650 -> 0.618266`.
- `reports/strict_generator_mlx_decode_eval_loop_statement_reachable_cap8_v1.json`
  and the wider `reports/strict_generator_mlx_decode_eval_loop_statement_reachable_cap8_wide_v1.json`
  are both `RED` with `0` loop candidates. This shows the refined objective
  removed the bad positive `continue` pressure but still did not create a
  replacement reachable update/finalizer path. The simple-return regression
  `reports/strict_generator_mlx_decode_eval_simple_return_loop_statement_reachable_regression_v1.json`
  remains `GREEN` with `8/8`.
- `reports/strict_generator_mlx_decode_eval_loop_statement_reachable_starvation_cap4_v1.json`
  adds top-level `split_decode_starvation` evidence. On the refined checkpoint,
  `4/4` loop tasks emit no candidates while the top beams remain
  `inside_loop_without_update` and `missing_local_return`; body previews show
  runaway nested expressions/conditions rather than a reachable update/finalizer
  path. This is no-credit diagnosis, not a candidate claim.
- `reports/strict_generator_mlx_private_adaptation_loop_statement_update_finalizer_v1.json`
  is `GREEN`: a role-filtered span objective boosts only admitted private
  `loop_body_update` and `top_level_finalizer` target spans, filtering out
  `177` decision spans. It improves heldout LM loss `0.669650 -> 0.610516`,
  weights `3,841` positions, and still reports `0` public rows, `0` external
  inference, and `0` candidate-generation credit.
- `reports/strict_generator_mlx_decode_eval_loop_statement_update_finalizer_cap8_v1.json`
  is `YELLOW`: the role-filtered checkpoint emits `2` loop candidates again.
  One candidate has a reachable update (`out.append(value)`) followed by a
  top-level finalizer (`return out`), so the zero-candidate wall moved. Verifier
  pass is still `0`; the residual is now semantic shallowness
  (`shallow_identity_accumulation`, missing numeric accumulation, missing list
  construction, missing windowed finalizer), and one candidate still contains
  an unreachable `continue` before update. The simple-return regression
  `reports/strict_generator_mlx_decode_eval_simple_return_loop_statement_update_finalizer_regression_v1.json`
  remains `GREEN` with `8/8`.
- `reports/strict_generator_mlx_private_adaptation_loop_semantic_operation_v1.json`
  is `YELLOW`: the new private semantic-operation span objective matched
  `192/192` admitted loop rows and weighted `4,533` operation/finalizer token
  positions across assignment transforms, augmented accumulator updates,
  projected append/add calls, collection removals, and transform finalizers.
  It explicitly excluded shallow direct loop-variable append/add spans from
  this semantic boost and improved heldout LM loss `0.610516 -> 0.565156`.
  The only failed gate is soft: the auxiliary plan head was already `1.0`
  accuracy and its loss moved slightly worse.
- `reports/strict_generator_mlx_decode_eval_loop_semantic_operation_default_after_flag_cap8_v1.json`
  is still `YELLOW`: default decode from that checkpoint emits `1` loop
  candidate, verifier pass remains `0`, and the surviving candidate is still
  `shallow_identity_accumulation` with missing list/windowed finalizer
  semantics. The simple-return regression
  `reports/strict_generator_mlx_decode_eval_simple_return_loop_semantic_operation_regression_v1.json`
  remains `GREEN` with `8/8`.
- `reports/strict_generator_mlx_decode_eval_loop_semantic_operation_no_shallow_append_cap8_v1.json`
  is `RED`: the opt-in `--block-shallow-loop-identity-update` decode hygiene
  experiment blocks the bad direct `accumulator.append(loop_var)` style but
  emits `0` loop candidates. This is useful negative evidence, not a default
  route: the stricter block is now explicit/off-by-default because the learned
  decoder has not yet learned an alternate semantic update path.
- `reports/strict_generator_mlx_decode_eval_loop_operation_hints_source_only_cap8_v1.json`
  is `RED`: reintroducing prompt-only operation hints into strict source text
  without adaptation keeps the source audit clean (`0` forbidden marker hits,
  `0` solution/test fragments, `7/8` prompt-operation-hint rows) but emits `0`
  loop candidates. This proves prompt hints are not immediately usable by the
  current checkpoint.
- `reports/strict_generator_mlx_private_adaptation_loop_operation_hints_v1.json`
  is `GREEN`: the same prompt-only operation-hint source contract trains
  cleanly. It improves heldout LM loss `0.601345 -> 0.507321`, improves the
  source-contrastive gap `0.488449 -> 0.565484`, keeps source audit clean, and
  uses `0` public rows, `0` external inference, and `0` candidate-generation
  credit. The decode report
  `reports/strict_generator_mlx_decode_eval_loop_operation_hints_cap8_v1.json`
  is still `YELLOW`: `1` loop candidate emits, pass remains `0`, and the
  residual is still shallow identity plus missing numeric/list/windowed
  semantics. Simple-return regression remains `GREEN` with `8/8`.
- `reports/strict_generator_mlx_private_adaptation_loop_slot_prefix_v1.json`
  and
  `reports/strict_generator_mlx_private_adaptation_loop_slot_prefix_low_v1.json`
  are both `GREEN` training runs for the new semantic-slot prefix objective.
  They weight exact admitted private target slots before `SLOT:BODY_START`
  (`1,525` positions across update/finalizer/guard/init/return-shape/loop-source
  roles), improve heldout LM loss, and keep no-cheat counters clean. Their
  loop decodes,
  `reports/strict_generator_mlx_decode_eval_loop_slot_prefix_cap8_v1.json` and
  `reports/strict_generator_mlx_decode_eval_loop_slot_prefix_low_cap8_v1.json`,
  are both `RED` with `0` loop candidates. Simple-return regressions remain
  `GREEN`. This is negative evidence that slot-prefix loss alone is not enough;
  it can make the prefix more confident while current body beam search still
  fails to realize a valid update/finalizer expression.
- `scripts/strict_generator_mlx_private_adaptation.py` now also supports
  decode-starvation negative replay. The selector consumes only
  `private_train_replay` top-beam body previews from strict decode reports,
  rebuilds prompt/signature source text from the original private row, audits
  that source text, and records the previews only as rejected bodies with
  `candidate_generation_credit=0`. The first combined replay run,
  `reports/strict_generator_mlx_private_adaptation_loop_starvation_replay_v1.json`,
  is `GREEN`: it selected `1` failed private candidate plus `8`
  decode-starvation beam previews, kept no-cheat counters clean, and improved
  heldout source-contrast gap `0.565484 -> 0.590117`. Decode from that
  checkpoint without extra progress constraints remains `RED` with `0` loop
  candidates, proving the training signal alone did not close loop blocks.
- `scripts/strict_generator_mlx_decode_eval.py` now has an opt-in
  `--enable-loop-progress-guard`. It is a task-blind search/progress
  constraint over generated plan-prefix state, not a renderer and not learned
  generation credit. With the identity blocker also enabled,
  `reports/strict_generator_mlx_decode_eval_loop_progress_guard_cap8_v1.json`
  remains `RED` with `0` loop candidates: blocking shallow append before the
  model has learned alternate expressions causes malformed update-expression
  search. With the progress guard but without identity blocking,
  `reports/strict_generator_mlx_decode_eval_loop_progress_guard_no_identity_block_cap8_v1.json`
  improves loadability: `12` integrity-verified transformer/hybrid loop
  candidates emit, parse/load cleanly, and the hard candidate-emission gate
  passes. Behavior remains `0/8` and all emitted loop candidates are shallow
  identity accumulations.
- The follow-up behavior replay
  `reports/strict_generator_mlx_private_adaptation_loop_behavior_replay_v1.json`
  consumes those `12` shallow identity candidates plus `4` remaining
  starvation prefixes as private negative/pairwise replay. It is `YELLOW` only
  on the soft auxiliary-plan-loss gate; replay integrity, public-training, and
  external-inference gates remain clean. The decode report
  `reports/strict_generator_mlx_decode_eval_loop_behavior_replay_progress_guard_cap8_v1.json`
  remains `YELLOW`: `12` loadable loop candidates, `0` behavior pass, `2`
  zero-candidate tasks, and unchanged residual labels
  `loop_without_decision_or_state_update`, `shallow_identity_accumulation`,
  `missing_gcd_call`, `missing_numeric_accumulation`,
  `missing_rle_branch_or_update`, and `missing_windowed_finalizer`.
  Simple-return regression
  `reports/strict_generator_mlx_decode_eval_simple_return_loop_behavior_replay_regression_v1.json`
  remains `GREEN` with `8/8`.
- A new private expression-synthesis objective now targets the actual update
  expression rather than the surrounding loop skeleton. The hook
  `loop_expression_synthesis_loss_boost` in
  `scripts/strict_generator_mlx_private_adaptation.py` extracts AST spans for
  loop update RHS/arguments and semantic top-level finalizer expressions, skips
  shallow direct loop-variable identity, matches those spans back into the
  encoded admitted private target, and reports `candidate_generation_credit=0`.
  `reports/strict_generator_mlx_private_adaptation_loop_expression_synthesis_v1.json`
  is `YELLOW` only because the already-saturated auxiliary plan loss ticked
  slightly worse; hard no-cheat gates pass. The expression hook matched
  `192/192` private loop rows and weighted `3,479` expression-token positions
  across `341` loop-update and `66` finalizer expression matches.
- Decode evidence is still negative on behavior. With the loop-progress guard
  and no identity block,
  `reports/strict_generator_mlx_decode_eval_loop_expression_synthesis_progress_guard_cap8_v1.json`
  stays `YELLOW`: `12` integrity-verified/loadable loop candidates, `0/8`
  behavior pass, and the same shallow-identity residual labels. The
  simple-return regression
  `reports/strict_generator_mlx_decode_eval_simple_return_loop_expression_synthesis_regression_v1.json`
  remains `GREEN` with `8/8`. With identity blocking enabled,
  `reports/strict_generator_mlx_decode_eval_loop_expression_synthesis_progress_guard_block_identity_cap8_v1.json`
  is `RED` with `0` emitted loop candidates. Its starvation traces show some
  beams now reach update calls/local returns before failing to close valid
  expressions and blocks, which narrows the next wall to expression/block
  closure and statement sequencing.
- The follow-up expression/block closure work is implemented and bounded as
  no-credit decode hygiene. `--enable-expression-closure-guard` uses generated
  prefix syntax only to close delimiters, end statements, dedent finished loop
  blocks, and start top-level local returns. With shallow identity blocking
  enabled, it moves the loop smoke from `0` candidates to
  `23` loadable/integrity-clean candidates in
  `reports/strict_generator_mlx_decode_eval_loop_expression_closure_guard_cap8_v1.json`,
  while behavior remains `0/8` and shallow identity drops to `2`. The
  simple-return regression stays `GREEN` at `8/8`.
- Closure replay training is clean but not behavior-solving.
  `reports/strict_generator_mlx_private_adaptation_loop_expression_closure_replay_v1.json`
  keeps private rows only, `0` public training rows, `0` external inference,
  clean source audits, and heldout LM improvement. Decode from it emits `26`
  loadable loop candidates in
  `reports/strict_generator_mlx_decode_eval_loop_expression_closure_replay_cap8_v1.json`
  and still scores `0/8`. The easy simple-return lane remains `GREEN` at
  `8/8`.
- `--enable-expression-value-guard` is now also implemented. It rejects
  generated-prefix-only expression value pathologies such as empty update-call
  arguments, direct set literals in append/update calls, and bare builtin
  objects closed as values. It uses no tests, solutions, public data, hidden
  labels, tools, or renderers, and records `candidate_generation_credit=0`.
  `reports/strict_generator_mlx_decode_eval_loop_expression_value_guard_cap8_v2.json`
  reduces junk loop candidates to `11` and produces cleaner calls such as
  `abs(value)`, but still scores `0/8` and starves several tasks. Its
  simple-return regression remains `GREEN` at `8/8`.
- A stronger private-only state-transition weighting run
  `reports/strict_generator_mlx_private_adaptation_loop_state_transition_v1.json`
  is `GREEN`: `256` private loop rows, `64` heldout replay rows, `0` public
  training rows, `0` external inference, no fallback/template/router credit,
  heldout LM improved, and weighted coverage over `180` loop decisions,
  `199` loop updates, `399` semantic loop updates, and `787` expression spans.
  Its decode
  `reports/strict_generator_mlx_decode_eval_loop_state_transition_cap8_v1.json`
  remains `YELLOW`: `14` loadable candidates, `0/8` behavior pass, still
  missing decision/state-update semantics, with some worse collection-mutation
  sequences. The simple-return regression
  `reports/strict_generator_mlx_decode_eval_simple_return_loop_state_transition_regression_v1.json`
  stays `GREEN` at `8/8`.
- A first learned state-transition prefix implementation now exists. The target
  tokenizer emits `SLOT:STATE_*` tokens for loop branch/update/finalizer shape
  before `SLOT:BODY_START`, and the decoder parses those slots into loop
  adequacy metadata without rendering Python or granting candidate-generation
  credit. The fresh state-slot MLX smoke
  `reports/strict_generator_mlx_pretraining_state_slots_smoke_v1.json` is
  `GREEN`: `14` state-slot vocab entries, heldout LM loss improved
  `8.192360 -> 4.241084`, `906,437` optimizer token positions, `0` public
  training rows, and `0` external inference calls.
- The private state-slot loop adaptation
  `reports/strict_generator_mlx_private_adaptation_state_slots_loop_v1.json`
  is also `GREEN`: `256` private loop rows, `1,882` weighted state-slot
  positions, hard no-cheat gates clean, and heldout LM improved. The behavior
  result remains negative but sharper. `reports/strict_generator_mlx_decode_eval_state_slots_loop_cap8_v1.json`
  still scored `0/8` and often skipped state slots. The required-state-slot
  decode `reports/strict_generator_mlx_decode_eval_state_slots_required_loop_cap8_v1.json`
  emitted state slots in every inspected prefix but stayed `0/8`. The category
  coverage decode
  `reports/strict_generator_mlx_decode_eval_state_slots_category_loop_cap8_v1.json`
  forced loop-source/init/update/state prefix coverage, kept `26` loadable
  candidates, and stayed `0/8`, with residuals narrowed mainly to wrong
  plan/operand selection rather than syntax/loadability. The simple-return
  regression
  `reports/strict_generator_mlx_decode_eval_state_slots_category_simple_return_regression_v1.json`
  remains `GREEN` at `8/8`.
- Causal operand-binding prefix targets are now implemented. The strict target
  builder emits `SLOT:BIND_*` tokens for loop/branch operand use, update
  operand use, and accumulator/finalizer binding; these are stripped before
  compilation, never render Python, and grant zero candidate-generation credit.
  The decoder parses those slots into loop adequacy metadata and filters
  impossible loop-source prefix slots against visible callable signature names.
- The binding-slot MLX smoke
  `reports/strict_generator_mlx_pretraining_operand_bindings_smoke_v1.json`
  is `GREEN`: target vocab `2889`, `14` binding slots, `14` state slots,
  `901,017` optimizer token positions, heldout LM loss improved
  `8.125375 -> 4.351802`, `0` public training rows, and `0` external inference
  calls.
- The loop-only binding adaptation
  `reports/strict_generator_mlx_private_adaptation_operand_bindings_loop_v1.json`
  is `GREEN`: `256` private loop rows, `1,909` weighted binding positions,
  `1,882` weighted state positions, hard no-cheat counters clean, and no
  fallback/template/router/tool credit. Its default loop decode
  `reports/strict_generator_mlx_decode_eval_operand_bindings_default_loop_cap8_v2.json`
  improves loadability over the fresh checkpoint (`6/8` integrity-verified and
  runtime-loaded attempts versus `0/8`) but remains `0/8` behavior. The
  explicit `--require-binding-prefix-groups` ablation
  `reports/strict_generator_mlx_decode_eval_operand_bindings_required_groups_loop_cap8_v2.json`
  regresses integrity to `4/8` and also remains `0/8`, so richer forced prefix
  coverage is not the next default route.
- The binding run does not preserve the easy lane. The loop-only checkpoint
  `reports/strict_generator_mlx_decode_eval_operand_bindings_simple_return_regression_v1.json`
  keeps `8/8` integrity/runtime load but scores `0/8` and emits inert loop
  bodies for simple-return rows. The balanced adaptation
  `reports/strict_generator_mlx_private_adaptation_operand_bindings_balanced_v1.json`
  is mechanically `GREEN` (`384` rows, `2,122` binding positions, `2,160`
  state positions), but
  `reports/strict_generator_mlx_decode_eval_operand_bindings_balanced_simple_return_v1.json`
  regresses to `0/8` integrity due syntax failures. This is negative evidence
  against more scalar prefix weighting or naive tier mixing.
- Specialist-head routing is now part of the canonical strict MLX decoder
  surface. `scripts/strict_generator_mlx_decode_eval.py` accepts
  `--specialist-checkpoint-report tier=report.json`, routes existing private
  replay tiers to selected learned checkpoints, and emits
  `strict_generator_private_replay_specialist_head_route_v1` receipts with
  zero candidate-generation credit. The route sees only tier choice and
  checkpoint reports; generation still sees strict prompt/signature text and
  the selected learned checkpoint, not tests, solutions, verifier labels,
  public payloads, answer templates, renderers, tools, or fallbacks.
- The first profiled specialist-head report
  `reports/strict_generator_mlx_decode_eval_specialist_heads_profiled_v1.json`
  is `YELLOW`: route profiles are enabled, `simple_return` is routed to
  `reports/strict_generator_mlx_private_adaptation_state_slots_loop_v1.json`,
  `loop_accumulate` is routed to
  `reports/strict_generator_mlx_private_adaptation_operand_bindings_loop_v1.json`,
  simple-return stays `4/4` intended-behavior passed, loop stays `0/4`
  behavior with `3` runtime-loaded attempts and `1` parse failure, and overall
  candidate integrity is `7/8`. This prevents the loop head from erasing the
  easy lane, but it does not solve loop semantics.
- The latest loop hygiene pass tightens expression-value and semantic-update
  evidence without claiming behavior. `scripts/strict_generator_mlx_decode_eval.py`
  now reports task-blind expression-value quality and can reject generated
  candidates that use bare builtin/type objects as runtime values or obvious
  invalid call argument types. `scripts/neural_seed_static_coherence.py` was
  corrected so valid multi-argument `max`/`min` and clamp-style
  `min(max(number, lo), hi)` bodies are not falsely rejected. The static
  return-dependency path also no longer treats builtin names used as callees,
  such as `int(digits)`, as data values. The findings packet
  `reports/strict_generator_mlx_value_guard_findings_v1.md` records the
  no-credit evidence summary. The clean
  adaptation report
  `reports/strict_generator_mlx_private_adaptation_value_guard_loop_v2.json`
  is `GREEN`: private rows only, `0` public training rows, `0` external
  inference, no fallback/template/router/tool credit, heldout LM loss improved
  `4.288703 -> 0.85482`, and corrected target static-guard pass rates are
  about `0.80` train / `0.79` heldout. The earlier
  `reports/strict_generator_mlx_private_adaptation_value_guard_loop_v1.json`
  is not clean capability evidence because it used the overbroad `min`/`max`
  guard.
- Decode still fails on the real behavior wall. The adapted checkpoint in
  `reports/strict_generator_mlx_decode_eval_value_guard_adapted_loop_v1.json`
  emits `6` integrity-clean loop candidates with `0/8` pass, but the bodies are
  shallow identity or self-append patterns. The stricter no-identity run
  `reports/strict_generator_mlx_decode_eval_value_guard_no_identity_loop_v2.json`
  emits `3` integrity-clean transformer/hybrid candidates and still scores
  `0/8`. Residuals are now precise: `missing_semantic_update_value=3`,
  `append_accumulator_to_itself=1`,
  `loop_without_decision_or_state_update=1`, and `missing_gcd_call=1`.
  The current wall is learned stateful update synthesis, not syntax,
  loadability, candidate integrity, or guard plumbing.
- Prompt-visible operation conditioning is now stricter and better
  instrumented, but remains no-claim evidence. `scripts/neural_seed_visible_source.py`
  now surfaces clamp/round operation tags from prompt/signature text, and
  `scripts/strict_generator_mlx_decode_plans.py` attaches those tags to
  source-condition adequacy plus operation-evidence residuals without using
  tests, solutions, public artifacts, teacher output, templates, renderers,
  tools, or fallback returns. The strict broad4 smoke
  `reports/strict_generator_mlx_decode_eval_prompt_operation_condition_strict_broad4_v3.json`
  emits `8` integrity-clean transformer/hybrid candidates, keeps public
  training rows, external inference, and fallback/template/router/tool credit
  at `0`, and reaches nontrivial-return rate `1.0`; behavior still remains
  `0/4`. The dominant residual is no longer emission starvation but missing
  learned operation/state-transition semantics: operation evidence is mostly
  absent, type handling fails `6` rows, wrong answers fail `2`, and
  body-action mismatches still include `missing_semantic_update_value`.
- The existing semantic-construction repair profile now includes source-condition
  operation-token internalization. `scripts/strict_generator_mlx_adaptation_weights.py`
  boosts only admitted private target tokens that satisfy prompt-visible
  operation tags, such as `min`, `max`, `round`, `sum`, `abs`, or arithmetic
  operators, and still grants `0` candidate-generation credit. The v2 private
  MLX smoke
  `reports/strict_generator_mlx_private_adaptation_source_condition_operation_internalization_smoke_v2.json`
  now covers every prompt-visible operation tag the source tagger emits:
  `op_abs_positive_filter`, `op_clip_to_range`, `op_gcd_reduce`,
  `op_numeric_summary`, `op_round_values`, and `op_windowed_delta`. It matched
  `93` source-condition rows, found `49` adequate private target rows, weighted
  `974` source-condition token positions including `116` operation-token
  positions, improved heldout LM loss, and kept public training rows, external
  inference, and fallback/template/router/tool credit at `0`. The follow-up
  broad4 decode
  `reports/strict_generator_mlx_decode_eval_source_condition_operation_internalization_broad4_v2.json`
  remains `YELLOW`: `7` generated candidates, `6` integrity-verified
  transformer/hybrid candidates, nontrivial-return rate `0.857143`, but
  behavior still `0/4` with `type_handling`, `wrong_answer`, and one
  compile/parse residual. This is training-path coverage evidence, not a
  promotion or capability claim.
- The expression-value guard now rejects a concrete generated-expression
  pathology that was polluting the Phase 10 candidate pool:
  `isinstance(<boolean/comparison expression>, type)`. This is implemented in
  `scripts/strict_generator_mlx_decode_guards.py`,
  `scripts/strict_generator_mlx_decode_plans.py`,
  `scripts/neural_seed_expression_value_guard.py`, and consumed by
  `scripts/strict_generator_mlx_decode_eval.py` as task-blind generated-prefix
  and post-decode hygiene. It does not render code, inspect tests/solutions,
  use public data, or grant learned-generation credit. The focused v3 smoke
  `reports/strict_generator_mlx_decode_eval_expression_value_isinstance_guard_broad4_v3.json`
  keeps public training rows, external inference, and
  fallback/template/router/tool credit at `0`; it records `16`
  `invalid_expression_value` guard rejections and the accepted candidate file
  has `0` `isinstance(data in ...)` style bodies. This is a guard repair, not a
  capability win: behavior remains `0/4`, generated candidates drop to `6`,
  integrity-verified candidates drop to `4/6`, nontrivial-return rate drops to
  `0.333333`, and no-function/syntax mismatches rise to `2`.

Non-claim: the new loop-plan exploration, adequacy checks, and body-action
traces are task-blind no-credit scaffolding. They do not use
tests/solutions/public artifacts and do not count as learned generation or
semantic capability.

Next concrete wall: learned loop-body action ordering and expression hygiene are
now better isolated, but the model remains semantic-shallow. Prompt-only
operation hints, semantic-slot prefix weighting, decode-starvation replay,
loop-progress guarding, expression-synthesis target weighting, expression
closure/value guards, state/binding slots, and specialist-head routing are all
implemented and audited. Together they recover loadable candidates and make
failures precise, but they still do not produce stateful semantic updates or
useful final returns. The next repair should keep the routed survival
architecture and improve the learned loop/update/finalizer specialist itself: a
larger/fresher transformer-hybrid checkpoint for stateful loops, a
verifier-positive operand-span/state-transition objective, or a loop-specific
head for statement sequencing that does not render fixed updates. Do not
reintroduce a fixed update renderer or count trace scaffolding/search
constraints as learned generation.
