# Theseus Training, Evaluation, And Benchmarking

2026-07-10 current data-authority note: source-level admission is no longer
sufficient for the canonical survival lane. Every selected training row must
match an `admit` receipt in
`runtime/data_governance/data_admission_receipts_v1.jsonl.gz`; the receipt
ledger is replayed against source hashes and the current contamination index.
`reports/training_data_lineage_audit.json` is `YELLOW` with `0` hard gaps:
`64,196` receipts, `63,892` admits, `304` heldout rejects, `9/9` adversarial
controls, five frozen lifecycle-policy simulations, and positive plus broken
11-kind deletion-closure fixtures. The warning is a `0.824771` recursive
synthetic share. These are data-governance results, not model quality,
forgetting, unlearning, learned-generation, or public-transfer evidence.

This document defines the local-only workflow for SymLiquid. For the current
live status, gate results, and next action, start with `docs/PROJECT_STATE.md`;
some older command examples below are retained as historical reference.

2026-06-06 local / 2026-06-07 UTC live note: the current Code LM path is
STS-default-on, and the
wide 160-task public calibration has been consumed exactly once and relocked.
That calibration scored `34/160 = 0.2125`: MBPP `3/32`, EvalPlus `4/32`,
BigCodeBench `21/32`, HumanEval `6/32`, and LiveCodeBench `0/32`, with
`0` regressions, `0` external inference calls, and no public tests or solutions
exported into training rows. The follow-up private residual repair v3 gate is
now `GREEN`: `88/120 = 0.733333` private heldout pass rate, no-admissible rate
`0.0`, LiveCodeBench-private stdin proxy `24/24`, and same-seed STS-on delta
`+0.633333` with `0` regressions. Evidence files are
`reports/private_residual_repair_v3_gate.json` and
`reports/private_residual_repair_v3_heldout_score.json`. This is private-only
repair evidence, not a new public score and not promotion evidence. The
remaining private weak lanes are return/interface fidelity, no-admissible
cleanup, and semantic ranker selection. The Mac sandbox path normalization issue
is fixed and validated by `reports/macos_sandbox_path_regression.json`, which
now covers the benchmark runtime helper, Code LM closure helper, and private
candidate verifier helper. Edge-contract-v2 private curriculum hygiene is also
fixed: `reports/edge_contract_v2_curriculum_contract_repair.json` is `GREEN`,
with `240/240` private generation plans complete and `0` unsafe public-training
rows. The current edge-v2 private closure and rescore are now GREEN:
`reports/edge_contract_v2_private_closure_runner.json` completed against the
repaired `240`-row curriculum, and
`reports/code_lm_closure_edge_contract_v2_private_rescore.json` independently
rescored the generated artifact. The Rust candidate artifact has `870` private
candidate rows with `224` labeled production
`sts_nonregression_union_candidate` fallbacks. Private heldout baseline is
`1/149 = 0.006711`, STS-off is `37/149 = 0.248322`, trained is
`41/149 = 0.275168`, private delta is `+0.268457`, STS repair delta is
`+0.026846`, and STS task-level regressions are `0`. Runtime load is
`0.987578`; public benchmarks were not rescored, public tests/solutions were
not used for training, and `external_inference_calls=0`. The successor private
public-transfer residual curriculum has since been completed as private-only
evidence; the current readiness packet is review-green, and public calibration
remains locked pending explicit operator approval.
Broad Private Generalization Ladder v1 is now the broader private repair
surface, and the learned-evidence gate has been refreshed under the tightened
no-body-memory rule:
`reports/broad_private_generalization_ladder_v1.json` is `GREEN` with `3000`
private train rows and `1008` private heldout rows across `12` families and
`24` categories. The latest full unattended refresh scored `1008/1008 = 1.0`
from `3696` candidate rows through private-train-induced broad semantic token
decoder candidates, with public tests/solutions unused and external inference
calls at `0`. The former broad-private STS semantic adapter remains demoted
diagnostic/non-promotion machinery. Public calibration stays locked.
Agent/tool-use transfer is now part of the current generalization evidence.
`reports/high_transfer_long_horizon_tool_use.json` is `GREEN` at `64/64`
private tool-use cases with `64` trace rows and `64` STS rows. The Mac-safe
runtime path defaults now use `scripts/theseus_runtime.py` instead of the old
Windows `D:/ProjectTheseus/...` paths. `reports/pufferlib4_rl_lane.json` is
`GREEN` through a local synthetic RL fallback while honestly reporting
Puffer-native/Ocean as missing: `54` policy trace rows, accuracy delta `+1.0`,
rollout reward delta `+1.32`, and
`policy_learning_backend=local_synthetic_rl`. `reports/cross_domain_sts_capsules.json`
is `GREEN` with `42` metadata-only capsules across conversation,
long-horizon tool-use, and RL policy lanes, named causal consumers, and
`external_inference_calls=0`. `reports/high_transfer_multi_turn_conversation_hard_v4.json`
is `GREEN`: hard-v4 conversation transfer graduates with `384/384` cases,
`943/943` turns, accuracy `0.9923456101190479`, and personality context ready
on every turn. `reports/agent_lane_transfer_gate.json` is still `YELLOW`
only because public transfer remains unresolved; its private/non-code transfer
requirements now pass: repo repair, terminal tool-use, RL policy learning,
cross-domain STS consumer effect, and hard-v4 conversation.
Mac-local unattended training readiness is now current as of 2026-06-07.
`scripts/overnight_learning_readiness.py` no longer treats the Windows
`D:/ProjectTheseus/...` runtime layout as mandatory on macOS/Linux; it checks
the configured runtime root, requires existing runtime paths, uses a `20 GiB`
non-Windows default free-space floor unless `THESEUS_MIN_RUNTIME_FREE_GIB` is
set, and still keeps the Windows `100 GiB` generated-artifact floor. The latest
`reports/overnight_learning_readiness.json` is `YELLOW` with
`overnight_launch_ready=true`, no red failures, public pass rate
`34/160 = 0.2125`, promotion blocked, and public calibration locked.
`reports/code_lm_closure_public_contract_preflight_seed23_32.json` is `GREEN`
with `0` external inference calls, no public tests/solutions used, and
`128` public task contracts checked; `reports/training_budget_plan.json` is
sufficient for the selected tool-agent/multi-turn frontier with `154112` train
environment steps. This is permission to keep private learning fed overnight,
not promotion evidence.
`reports/broad_private_learned_distillation_gate_v1.json` is now `GREEN` under
the stricter no-body-memory rule. Learned-only and strict novel learned-only
broad-private scoring both pass `1008/1008 = 1.0`; strict-novel candidate rows
are `3360`; exact private train-body memory pass count is `0`; prototype pass
count is `0`; pass AST shape count is `24`; pass normalized AST unique count is
`24`; top duplicate rate is `0.041667`; and train body overlap is `0.0`. This
closes the stale replay blocker for this lane. It remains private synthetic
transfer evidence only, not public promotion evidence.
`reports/broad_private_semantic_alias_gate_v1.json` is now `GREEN`: a private
semantic-alias stress gate rewrites all `1008` heldout rows across all `24`
broad-private categories so exact semantic-family lookup cannot be reused.
Learned-only alias candidates pass `1008/1008 = 1.0`, inferred token passes
are `1008`, candidate rows are `3696`, learned-only candidate rows are `3360`,
diagnostic-adapter passes are `0`, prototype passes are `0`, exact semantic key
reuse is `0`, public candidate sidecars are empty, and external inference calls
are `0`. This is stronger private transfer evidence than the plain broad
ladder, but it is still private-only and does not change the locked public
calibration state.
`reports/broad_private_novel_composition_gate_v1.json` is now `GREEN`: the
private novel-composition stress gate was refreshed after the decoder
argument-role fingerprint repair. It writes a full-scale `1008` heldout rows
across `6` two-step composition specs and forces reusable private-train token
bodies to be composed rather than selecting one exact semantic family.
Composition-only candidates pass `1008/1008 = 1.0`, composition token pass
count is `1008`, candidate rows are `4032`, composition candidate rows are
`1008`, diagnostic-adapter passes are `0`, prototype passes are `0`, public
candidate sidecars are empty, and external inference calls are `0`. The
refreshed run took `550.334s`, with `375.780s` in STS-on fanout and `168.682s`
in STS-off control. This is private-only transfer evidence, not public
promotion evidence.
`reports/private_unseen_transfer_challenge_v1.json` is now `GREEN`: a private
OOD transfer challenge rewrites `240` v5 heldout rows so exact semantic-family
keys cannot be replayed. After the train-novel decoder repair for
project-progress and room-capability aliases, STS-on scores `240/240 = 1.0`;
learned-only also scores `240/240 = 1.0`; same-seed STS-off control is
`0/240`; exact semantic key replay count is `0`; prototype pass count is `0`;
public-data leakage hits are `0`; and external inference calls are `0`.
`reports/private_unseen_transfer_challenge_v1_learned_distillation_gate.json`
is also `GREEN`: exact private-train normalized body overlap is `0.0`,
normalized AST overlap is `0.0`, pass AST shape count is `12`, and top pass
duplicate rate is `0.108333`. This is stronger private transfer evidence, but
it is still private-only and does not unlock public calibration.
The corrected full broad-ladder refresh completed in `570.937s`, with
`387.978s` in STS-on fanout and `180.750s` in STS-off control. The earlier
stale-runtime concern is no longer the active blocker for this lane.
`reports/post_distillation_public_transfer_readiness_v1.json` is `YELLOW`:
freshness and integrity gates are clean and the operator lock is active, but
public transfer readiness remains blocked by the already-spent wide public score
`34/160 = 0.2125` below the `0.70` floor, with all five public cards below
floor. Do not use the older May 29 or older 32-task scores below as current
promotion evidence.
`reports/public_calibration_readiness_packet.json` is `GREEN` in
`post_distillation_v4_operator_review` mode for technical operator review. It
now uses the current pre-public audit (`operator_review_ready=true`,
`learned_token_pass_count_total=8752`) and exhausted frontier expander
(`no_private_frontier_action_remaining`) as the private readiness handoff. This
is not promotion evidence: the already-spent wide public score remains
`34/160 = 0.2125`, below the `0.70` floor. The pinned public surface still has
`160` task IDs and no public leak hits, and `public_calibration_allowed=false`
because the operator lock remains active. The packet does not run calibration,
and no `post_v4_seed23_5x32` public result exists.
`reports/pre_public_generalization_readiness_audit.json` is the current
machine-readable pre-public handoff. It consolidates the governor, private
residual ratchet, agent-lane gate, teacher preflight, and public-boundary
invariants. Current state is `YELLOW`: private code transfer, private agent
transfer, teacher path, and hard safety gates are ready, but
`public_transfer_floor_cleared` is false. The frontier expander executed
`expand_private_unseen_transfer_challenge_240`,
`expand_private_residual_frontier_1008`,
`expand_private_unseen_transfer_challenge_360`, and
`expand_private_residual_frontier_1344`, then completed the sharded
`expand_private_residual_frontier_840_spec21` and
`expand_private_residual_frontier_1040_spec26` proofs successfully. It now reports
`no_private_frontier_action_remaining`, so the pre-public audit queues only the
locked `operator_review_bounded_public_calibration_locked` item and still keeps
`public_calibration_allowed=false`.
`reports/operator_bounded_public_calibration_dry_run.json` is `GREEN`: the
guarded runner dry-run verifies the exact one-shot command shape, confirms the
operator lock before and after the dry-run, confirms the proposed output
artifact is absent, and does not execute public calibration. An explicit
operator approval file is still required before `--execute`.
The v3 private-public-transfer successor has now been rerun after the
strict-novel token decoder repair:
`reports/edge_contract_v3_verifier_mismatch_public_transfer_heldout_v3_contract_strict_syntaxrepair_full192_broad_score.json`
scores `192/192 = 1.0` across the full private heldout. All six families pass
`32/32`: algorithmic planning, stateful verifier mismatch, no-admissible
interface coverage, return-shape contract, stdin public-transfer proxy, and
metamorphic verifier mismatch. The refreshed same-seed STS-off control passes
`126/192 = 0.65625`, so STS delta is `+0.34375` with `0` regressions and the
score is now `GREEN`.
`reports/edge_contract_v3_verifier_mismatch_public_transfer_v3_contract_strict_syntaxrepair_full192_broad_learned_distillation_gate.json`
is `GREEN`: learned-only and strict-novel learned-only candidates pass
`192/192 = 1.0`, learned-token pass count is `192`, learned-only candidate rows
are `274`, exact train-body memory candidate rows are `0`, exact train-body
memory pass count is `0`, prototype pass count is `0`, diagnostic-adapter pass
count is `0`, public tests/solutions are not used, external inference calls are
`0`, and source/release freshness is proven. All v3 passes come through
`rust_code_lm_edge_contract_v3_strict_novel_token_decoder_v1_sts_conditioned`
modes, with `14` pass AST shapes and `14` unique normalized pass ASTs. This is
private learned-transfer evidence, not public promotion evidence.
The v4 public-safe broad transfer maturity curriculum is now full-gate proven
as private learned-token transfer evidence with corrected full causal
STS-delta evidence:
`reports/public_safe_broad_transfer_maturity_v4.json` is `GREEN` with `3000`
private train rows, `1008` private heldout rows, `6` maturity families, `24`
categories, `0` private solution failures, `0` public-data leakage hits, and
`0` external inference calls.
`reports/public_safe_broad_transfer_maturity_v4_smoke64_score.json` is
`GREEN`: corrected STS-on smoke passes `64/64 = 1.0`, same-seed STS-off passes
`0/64`, STS delta is `+1.0`, and regressions are `0`. The source fixes preserve
four-argument contracts, preserve the `extra` tuple-rest alias for
multi-argument generated bodies, stop the STS control policy from creating
streams when the control STS file is empty, and add explicit private-safe task
STS streams with `scripts/private_task_sts_streams.py`.
`reports/public_safe_broad_transfer_maturity_v4_smoke64_learned_distillation_gate.json`
is now `GREEN`: learned-only v4 smoke passes `64/64 = 1.0`, with prototype
pass count `0` and prototype rows `0`.
`reports/public_safe_broad_transfer_maturity_v4_score.json` is `GREEN`: full
v4 heldout passes `1008/1008 = 1.0` across all `6` maturity families and `24`
categories from `3696` candidate rows. The corrected same-seed STS-off control
has `1008` rows across `1008` tasks, `0` STS-conditioned rows, `0`
decoder-control-policy applications, and scores `0/1008`, so STS delta is
`+1.0`; no-admissible rate is `0.0`, regressions are `0`, and no public tests,
public solutions, or external inference are used.
`reports/public_safe_broad_transfer_maturity_v4_learned_distillation_gate.json`
is `GREEN`: learned-only v4 heldout also passes `1008/1008 = 1.0`, learned-token
pass count is `1008`, learned-only candidate rows are `3444`, prototype pass
count is `0`, prototype rows are `0`, and verifier-admissible pass count is
`1008`. The refreshed learned-maturity gate also proves source/release
freshness, normalized AST diversity (`24` pass shapes), control-structure
coverage, and private-train novelty: exact normalized AST/body train-overlap
rates are both `0.0`. Public calibration remains locked; the current governor
queue still stops at locked operator review, and the pre-public audit now records
that no configured private frontier action remains. Public calibration still
requires the operator to approve exactly one bounded public calibration, then
relock immediately and audit broad public transfer.
The active private-only follow-up is now Private Ecology Generalization v5:
`reports/private_ecology_generalization_v5.json` is `GREEN` with `1800` private
train rows and `720` private heldout rows across project memory, tool
transcripts, file/storage manifests, device routing, long-horizon plans, and
spatial/operator workflows. Train and heldout private solution failures are
`0`, public-data leakage hits are `0`, and external inference calls are `0`.
The queue for the next private autopilot step is
`reports/private_ecology_generalization_v5_queue.jsonl`. The Rust decoder
contract recognizer and private-train induced token loader now accept
`project_theseus_decoder_contract_v5_private_ecology_generalization`; focused
local proof passed: `cargo test -p symliquid-cli private_train_prototype -- --nocapture`.
`reports/private_ecology_generalization_v5_refresh.json` is now the canonical
full-refresh proof for the v5 lane. It is `GREEN`: the runner regenerated v5
rows, wrote `720` private-safe STS streams, reran STS-on fanout, reran
same-seed STS-off control, rescored the full `720` heldout set, reran the
learned-only gate, and proved score/learned artifacts are fresh after the
regenerated curriculum. The latest run took `664.487s`, emitted `2760`
STS-on candidates, `720` STS-off control candidates, `0` public candidates,
and used `0` external inference calls.
`reports/private_ecology_generalization_v5_smoke72_score.json` is now `GREEN`:
the private-only v5 smoke passes `72/72 = 1.0` across all `6` workflow families
and `12` categories, same-seed STS-off control is `0/72`, STS delta is `+1.0`,
no-admissible rate is `0.0`, regressions are `0`, public-data leakage hits are
`0`, and external inference calls are `0`.
`reports/private_ecology_generalization_v5_smoke72_learned_distillation_gate.json`
is also `GREEN`: learned-only candidates pass `72/72 = 1.0`, learned-token pass
count is `72`, learned-only candidate rows are `276`, prototype pass count is
`0`, prototype rows are `0`, verifier-admissible pass count is `72`, public
tests/solutions are not used, external inference calls are `0`, and source /
release-binary freshness is proven.
`reports/private_ecology_generalization_v5_full480_score.json` is now `GREEN`:
despite the historical `full480` filename, the current canonical refresh passes
`720/720 = 1.0` across all `6` workflow families and `12` categories.
Same-seed STS-off control is `0/720`,
STS delta is `+1.0`, no-admissible rate is `0.0`, regressions are `0`,
public-data leakage hits are `0`, and external inference calls are `0`.
`reports/private_ecology_generalization_v5_full480_learned_distillation_gate.json`
is also `GREEN`: learned-only full v5 passes `720/720 = 1.0`, learned-token
pass count is `720`, learned-only candidate rows are `2760`, prototype pass
count is `0`, prototype rows are `0`, verifier-admissible pass count is `720`,
public tests/solutions are not used, external inference calls are `0`, and
source/release-binary freshness is proven. This v5 lane is private generated
ecology pressure, not a public score, not teacher-applied training, and not
promotion evidence.
The active post-v4 shadow lane now adds a narrower public-residual proxy without
spending or leaking another public calibration:
`reports/post_v4_private_shadow_transfer_v1.json` is `GREEN` with `12000`
private train rows and `2400` private heldout rows across `6` abstract residual
families and `24` private categories. It consumes only aggregate residual
summary labels from the already-spent public calibration and writes no public
benchmark prompts, tests, solutions, traces, score labels, task IDs, or
benchmark names into generated rows. Leakage hits are `0`, private solution
failures are `0`, and external inference calls are `0`.
`reports/post_v4_private_shadow_transfer_v1_smoke160_score.json` is `GREEN`:
STS-on private shadow heldout passes `2400/2400 = 1.0`; same-seed STS-off control
passes `0/2400`, STS delta is `+1.0`, regressions are `0`, no-admissible task
rate is `0.0`, and no public tests/solutions or external inference are used.
`reports/post_v4_private_shadow_transfer_v1_smoke160_learned_distillation_gate.json`
is also `GREEN`: learned-only private shadow heldout passes `2400/2400 = 1.0`,
learned-token pass count is `2400`, prototype pass count is `0`, prototype rows
are `0`, verifier-admissible pass count is `2400`, and source/release-binary
freshness is proven. This clears the post-v4 private shadow
prototype-dependency check, but it is still private-shadow evidence. The public
wide score remains `34/160 = 0.2125`, public calibration remains locked, and
model promotion/growth remain blocked until an operator-approved bounded public
run is justified by the readiness protocol.
`scripts/post_v4_generalization_autopilot_v1.py` is now the current unattended
private runner for this lane. The latest executed report,
`reports/post_v4_generalization_autopilot_v1.json`, is `GREEN`: it regenerated
`12000` private train rows and `2400` private heldout rows, rebuilt `2400`
private-safe STS streams, reran STS-on fanout with the private beam-precompute
policy disabled, reran same-seed STS-off control, rescored the heldout set at
`2400/2400 = 1.0`, reran the learned-only gate at `2400/2400 = 1.0`, and wrote
heartbeat, ledger, queue, and scale-specific archive artifacts.
Prototype pass count remains `0`, STS-off control remains `0/2400`, and public
calibration remains locked. The latest ratchet-executed run took `388.063s`;
the archive
directories for `480`, `960`, `1440`, `1920`, and `2400` heldout runs all
report `0` copy failures.
`reports/post_v4_generalization_autopilot_v1_scaling_profile.json` is also
`GREEN`: it preserves measured timing for all five scale runs. The latest run
spent `370.141s` total in fanout, including `279.303s` in STS-on fanout and
`90.838s` in STS-off control. The configured `2400` heldout cap is reached, so
the queue now records cap-reached plus readiness/operator-review items instead
of scheduling another `2400 -> 2400` private run. This confirms the runtime-policy fix reopened private scaling: the
older `1440` run spent `1721.936s` total in fanout before the policy fix. The
private loop is closer to self-sufficient evidence generation; it does not
change the public score or unlock public calibration.
`reports/theseus_generalization_governor_v1.json` is the current private-only
control surface for the ASI-relevant transfer wall. It is `YELLOW`, not because
of a safety failure, but because the already-spent wide public calibration is
still `34/160 = 0.2125`, below the promotion floor. Hard gates are clear:
`reports/public_calibration_operator_lock.flag` is active,
`public_calibration_allowed=false`, forbidden post-v4 public artifacts are
absent, the dry-run did not execute public calibration, no public tests or
solutions were used for training, and external inference calls are `0`. The
governor now sees broad-private, post-v4 shadow, v5 private ecology,
semantic-alias private transfer, novel-composition private transfer, private
unseen-transfer challenge, private residual-frontier transfer, architecture
guidance, architecture experiment governance, teacher preflight, causal
architecture-delta, and student-first evidence hygiene as current private
evidence.
The current causal architecture-delta proof is a
`24`-task private same-seed run with private heldout pass-rate delta
`+0.208333`, no-admissible-rate delta `-0.208333`, private receiver eligibility
delta `+0.291667`, private semantic test delta `+0.666667`, `4` positive
semantic families, `0` semantic family regressions, `0` public tasks, and no
public tests/solutions. The governor also now requires broad-private learned
pass-path structural maturity before counting `private_broad_green`: `24`
normalized AST families, `24` AST shapes, top normalized duplicate rate
`0.041667`, control-structure coverage ready, and exact train-overlap rates
`0.0`.
Semantic-alias learned-only transfer is now full-size at `1008/1008 = 1.0`
with `0` prototype/diagnostic-adapter passes; novel-composition transfer is
`1008/1008 = 1.0` composition-only with `0` prototype/diagnostic-adapter
passes; private unseen-transfer learned-only transfer is
`360/360 = 1.0` with `0` exact semantic replay, `0` prototype passes, and
`0.0` normalized train-body overlap; and private residual-frontier transfer is
`GREEN` with full STS-on pass `1040/1040 = 1.0` across `26` private residual
specs, frontier-token passes `1000/1040 = 0.961538`, STS-off control
`60/1040 = 0.057692`, `0` prototype/diagnostic-adapter passes, empty public
candidate sidecars, and external inference calls `0`. Learned-token pass count
total is now `8752`. Contract-blind private transfer is also `GREEN`: `240`
heldout rows have semantic names withheld, strict learned-only transfer passes
`240/240 = 1.0`, STS-off control is `0/240`, prototype and diagnostic-adapter
pass counts are both `0`, body-memory replay candidate rows are `0`, and
external inference calls are `0`. The governor now requires the matching
contract-blind learned-maturity gate too: decoder source/release freshness is
true, pass-path diversity is `36` AST shapes and `36` unique normalized ASTs,
top duplicate rate is `0.029167`, control-structure coverage is ready, and
exact normalized AST/body train overlap is `0.0`.
`reports/private_residual_self_improvement_ratchet_v1.json` now converts the
spent public residual summary into a private-only self-improvement queue. It is
`YELLOW` because the decision is `retry_private`, not because a hard gate
failed: the operator lock is active, forbidden post-v4 public artifacts are
absent, public prompts/tests/solutions are not embedded, teacher apply remains
blocked, and external inference calls are `0`. The ratchet executed
`run_private_residual_shadow_autopilot` and
`refresh_semantic_alias_transfer_gate` and
`refresh_novel_composition_transfer_gate` and `refresh_private_ecology_v5` and
`teacher_preflight_proposal_only_no_live` and `refresh_generalization_governor`
successfully. All `6` queue items are now complete, so the governor no longer
queues a no-op ratchet execution.
`reports/private_residual_frontier_v1.json` is the latest private-only frontier
gate after the ratchet queue is exhausted. It reads only aggregate public
residual categories and turns `verifier_mismatch`,
`no_admissible_candidate_regression`, `return_shape`, and
`algorithmic_planning` into `26` harder private composition specs and `1040`
heldout rows. The current run uses sharded fanout (`10` shards of `104` rows)
and aggregate scoring, so long private frontier work is now diagnosable and
safer to run unattended. Full STS-on scoring passes `1040/1040 = 1.0`,
frontier-token candidates pass `1000/1040 = 0.961538`, STS-off control is
`60/1040 = 0.057692`, no-admissible rate is `0`, diagnostic-adapter passes are
`0`, prototype passes are `0`, public candidate manifests are empty, and
external inference calls are `0`. The expanded gate now includes graph
components, shortest hops, string-DP LCS length, balanced parentheses, and
record grouping; a one-step learned-token composition bug was fixed so
single-step graph/grouping specs are not duplicated into invalid `*_then_*`
modes. The v4, post-v4 shadow, full v5, and unseen-transfer private lanes now
clear fresh structural/train-novelty gates; the remaining wall is honest public
transfer below floor, so another public review decision still requires explicit
operator approval.
`reports/code_lm_closure_rust_post_v4_runtime_breakdown_smoke4_fanout.json`
is the focused runtime-instrumentation proof: the 4-task private-only smoke is
`GREEN`, emits `candidate_fanout_runtime_breakdown`, separates shared precompute
wall time from per-task generation wall time, emits `16` private candidates,
emits `0` public candidates, and fixes shared-precompute accounting so it is no
longer summed as per-task verifier/cache work.
`reports/post_v4_fanout_precompute_ablation_v1.json` is now the bounded
runtime-policy proof: the 64-task private-only A/B probe keeps both default and
`THESEUS_CODE_LM_BATCHED_BEAM_CACHE=0` variants at `64/64 = 1.0`, emits `0`
public candidates, and cuts fanout runtime from `32049ms` to `8496ms`
(`0.734906` runtime delta rate). The post-v4 autopilot may use this private
runtime evidence to disable batched beam precompute for the next private scale
probe; it is not promotion evidence and does not unlock public calibration.

## Hard Rule

```text
SymLiquid evaluation must not call external proprietary models.
```

Public scores from other systems may be cited as metadata, but we do not rerun
their inference. We train and evaluate our own model on the same benchmark
contracts where possible.

## Current Code LM Fanout Contract

Current state, 2026-05-29:

- Normal Code LM recovery uses train-once checkpoint fanout, not the old
  monolithic closure and not repeated-training shards.
- `scripts/code_lm_train_once_fanout.py` now emits source, release-binary,
  checkpoint, fanout-report, and candidate-manifest provenance. Fanout reports
  are diagnostic-only when they are older than the source fingerprint or release
  binary they name.
- The current bounded private-only scale smoke is
  `reports/code_lm_fanout_fast_path_scale_smoke64_cpt4_private_only.json`.
  It reuses
  `reports/student_code_lm_checkpoint_private_pressure_private_recovery_train_once_fanout_v1.json`,
  runs 64 private eval tasks with `candidates_per_task=4`, emits zero public
  tasks, performs no retraining, and is GREEN.
- The private 64-task smoke dropped from `159753ms` to `7128ms` after private
  fanout up to four candidates was routed through the low-latency lazy path.
  After bounding private-only low-latency comparator/prototype work, the
  refreshed smoke is GREEN at `6041ms`. Private candidate expansion is now
  `2432ms` (`2458ms` including artifact write), or `38.406ms/task` wall-clock,
  with `175` private candidates and `163` token-level candidates.
- The canonical train-once fanout refresh now reuses the existing checkpoint
  and current release binary without retraining. Two public-metadata speed
  fixes are now proven: first, the STS bridge stopped paying the full recursive
  fallback when the parallel contract/prompt families were already
  candidate-rich; then public metadata fanout started using the low-latency
  contract path and exiting after the first accepted public metadata candidate.
  A later private-only prototype/comparator trim is explicitly disabled for
  public metadata so the public calibration boundary keeps the richer clean
  path. Wrapper wall time dropped from `244774ms` to `65337ms`, then
  `18333ms`, and now to `14444ms`.
  Public candidate expression work dropped from the historical `333057ms`
  cumulative branch cost to `3820ms` while preserving 32/32 public task
  coverage, `68` token-level learned candidates, `68` verifier/guardrail
  passes, and `0` template-like candidates. The refreshed artifacts are GREEN
  and current against source, binary, and checkpoint provenance.
- Fanout reports now split top-level timing into task-manifest load,
  STS-conditioning load, checkpoint model load, candidate expansion, artifact
  write, ranker/prefilter/verifier-cache summary, and filter/gate reporting.
  Candidate rows also expose per-task phase categories for candidate expansion,
  STS conditioning, ranker/prefilter, verifier/cache, and other runtime.
- Public calibration and model growth remain locked. Private speed smokes are
  runtime evidence only; they do not prove public transfer or promotion.
- Watchdog/system-efficiency refreshes now update cheap upstream evidence
  first: resource governor, Hive scheduler worker-plan, then performance
  optimizer, then system efficiency. This prevents stale resource-throttle or
  incomplete-worker-plan rows from becoming fake overnight goals. On
  2026-05-29, this cleared the stale performance optimizer bottlenecks:
  `performance_optimizer` is GREEN with `0` bottlenecks. A follow-up audit
  classification now moves superseded historical Code LM wall-clock rows into
  `stale_control_debt` instead of active `loop_bottlenecks`: the current
  `system_efficiency_audit` is YELLOW with `3` findings, `0` active loop
  bottlenecks, and `1` stale-control-debt row. Stale train-once full-manifest
  source evidence is no longer an overnight work target when a bounded
  current-source smoke is fresh.
  The optimizer also records
  control-plane freshness, so an older Hive scheduler report cannot create a
  fake incomplete-worker-plan wall after a newer resource-governor decision.
  After refreshing resource-governor and scheduler evidence on 2026-05-29,
  `hive_scheduler` planned `5` worker chunks, including the required local
  CUDA eval, CUDA readout-training, and CUDA rollout chunks, and
  `performance_optimizer` was GREEN with `0` bottlenecks.
  `resource_governor.py` also separates true profile training from ratchet
  maintenance status, so `refreshing_ratchet` no longer masquerades as an idle
  GPU training job.
  The remaining efficiency work
  is now maintainability cleanup, duplicate-service launch hygiene, root-disk
  artifact placement, and broad public-transfer quality rather than old
  checkpoint timing debt.
- Dashboard singleton service starts now use a process-level guard, not only
  the dashboard's in-memory job table. If another launcher already has
  `sparkstream_daemon.py`, `hive_node.py daemon`, or `hive_relay.py` running,
  the dashboard start path returns `already_running` with the existing PIDs
  instead of spawning another service. `service_process_hygiene` now reports
  these duplicate-prevention guards separately; on 2026-05-29 it showed
  `3/3` guarded launch paths while still honestly reporting the three live
  duplicate services. This does not kill current duplicates; it blocks future
  duplicate launch waste without hiding the service hygiene warning.
- Watchdog integrity checks now separate stale verifier reports from actual
  hard verifier failures. A stale `grammar_suckers` report with
  `python_invalid_promotion_eligible_count=0` stays YELLOW freshness debt
  instead of becoming a RED malformed-candidate claim, and a stale
  `deterministic_taming_stack` report with `hard_failure_count=0` stays YELLOW
  instead of RED. After this fix and a targeted personality-runtime refresh,
  the watchdog was YELLOW with only the live duplicate-service warning; grammar
  invalid promotion candidates remained `0` and deterministic hard failures
  remained `0`. The integrity classifiers live in
  `scripts/autonomy_watchdog_helpers.py` so the watchdog stays orchestration
  code instead of growing another hard maintainability hotspot.
- Overnight goals should be architecture-improvement goals, not report loops:
  name one subsystem, one bottleneck or wall, one source-level change class,
  one verification command/report, and one stop condition. If the next wake
  cannot produce a source change, measured runtime delta, cleaner evidence
  contract, or a blocked-waste proof, it should not spend the night repeating
  the same checks.
- A good all-night goal right now is therefore specific: split one high-impact
  maintainability hotspot that sits on an active path, preserve its public
  behavior, and prove the same bounded smoke/audit evidence still passes. Do
  not set a goal whose only expected output is refreshed reports or repeated
  automation setup.
- Hive node maintainability cleanup on 2026-05-29 split peer registry, slot
  state, update API, operator runtime summaries, relay/coordinator sync,
  cross-node storage helpers, and operator target/accelerator summaries out of
  `scripts/hive_node.py`. The node file is now `1783` lines, below the Python
  soft limit, and system efficiency reports `4` maintainability hotspots,
  `0` hard hotspots, and maintainability score `0.900`. Verification used
  `py_compile`, `hive_node.py probe`, and `operator_status`; no training or
  public calibration was launched for this cleanup.
- High-transfer scheduler maintainability cleanup on 2026-05-29 split the
  former `3096`-line `scripts/high_transfer_curriculum_scheduler.py` into a
  `116`-line CLI wrapper plus focused modules:
  `high_transfer_scheduler_common.py`, `high_transfer_scheduler_builder.py`,
  `high_transfer_scheduler_code_state.py`, and
  `high_transfer_scheduler_rotation.py`. Pre/post scheduler smokes preserved
  `39` concepts, `19` ready tasks, `10` critical tasks, `421` donor/receiver
  checks, identical concept status/priority/command signatures, and identical
  task IDs/commands. After the split, `system_efficiency_audit` reports `3`
  maintainability hotspots, `0` hard hotspots, and maintainability score
  `0.925`.
- Code LM public fanout now admits the normal `candidates_per_task=8` setting
  into the public metadata low-latency path. Before this fix, the same bounded
  32-public-task smoke fell through to the expensive template-free parallel
  family path and spent `153363ms` in public candidate generation. After the
  gate fix, `reports/code_lm_fanout_public_low_latency_limit8_smoke32.json`
  completed GREEN with `5868ms` public generation time, `32/32` task coverage,
  `68` public candidate rows, and `0` template-like candidates. This is
  runtime/control evidence only; public calibration and model growth remain
  locked.
- The same-seed non-STS comparator now uses lazy fallback: it tries the cheap
  visible-contract comparator first and pays prototype transduction only when
  the diagnostic comparator row is still missing. The public low-latency path
  also skips shared batched-beam precompute when single-accepted public
  metadata fanout can exit before beams are used. On the same 32-task smoke,
  `reports/code_lm_fanout_lazy_beam_precompute_skip_smoke32.json` completed
  GREEN with public generation reduced to `801ms`, the same `68` public
  candidate rows, `30` same-seed comparator rows, and `0` template-like
  candidates.
- State-sequence decode now precomputes static prompt/contract scores once per
  task and scores only dynamic beam features inside the completion loop. This
  preserves the decoder contract while removing repeated static feature
  construction from the fanout hot path.
- Fanout-only refresh is now bounded by default. `--refresh-fanout-only` writes
  sidecar current-source smoke artifacts and refuses to masquerade them as the
  canonical full closure manifests; a full canonical fanout refresh now requires
  the explicit `--full-fanout-refresh` flag. On 2026-05-29, the bounded
  sidecar refresh reused the existing checkpoint, emitted
  `reports/code_lm_closure_rust_private_pressure_private_recovery_train_once_fanout_v1_current_source_smoke_fanout.json`,
  completed in `4s`, generated `1` private candidate row and `2` public
  metadata candidate rows, and recorded `18ms` public candidate generation.
  This is speed/freshness evidence only, not promotion evidence and not public
  calibration.

## Current Code Learning State

Current state, 2026-05-29:

```text
active family: coding_local_sandbox
best clean public calibration card: source_human_eval, 32 tasks, pass rate 0.78125
below-floor receiver cards: MBPP 0.6875, EvalPlus 0.59375, BigCodeBench 0.25, LiveCodeBench 0.21875
broad matrix: 160 public calibration tasks across HumanEval/MBPP/EvalPlus/BigCodeBench/LiveCodeBench, best-clean-per-card aggregate pass rate 0.50625
latest board-executed receiver calibration: 49/128 = 0.382812 across MBPP/EvalPlus/BigCodeBench/LiveCodeBench, clean no-leakage/no-template/no-external-inference gates
next rotation target: private source-agnostic edge/type/interface/algorithmic pressure selected by the high-transfer scheduler; public recalibration should wait for a decoder/generator source change or stronger private gate evidence
candidate gate: promote=false
real public code calibration: broad matrix YELLOW; EvalPlus, BigCodeBench, and LiveCodeBench below promotion floor
transfer generalization audit: YELLOW; only HumanEval is above floor, aggregate broad pass rate is 0.50625, cross-card spread is 0.5625, and shared targets are type/return-shape, edge conditions, and admissibility/interface, with BigCodeBench-heavy algorithmic planning pressure
required promotion floor: 0.70
candidate source: student_code_lm_checkpoint_v1
score semantics: calibration, not broad public mastery
token-level learned generation: true
template-like benchmark candidates: 0
loop-closure benchmark candidates: 0
public expression-fallback passes: 0
STS native parallel generation: wired; positive but weak public causal signal
BigCodeBench/LiveCodeBench: real D:-staged public tasks with clean task-matched candidates; both have 32-task clean slices and remain below floor. Latest best-clean-per-card matrix has BigCodeBench 8/32 and LiveCodeBench 7/32
latest algorithmic planning pressure: 960 private generated rows, 0 private solution failures, interval/window/frequency/graph/state/two-pointer families
latest algorithmic STS patch: STS affects skeleton admission/ranking directly; public smoke STS delta stayed positive, and the current source-agnostic private closure shows private STS repair delta +0.05021 with 0 regressions
latest typed-interface pressure: 960 private generated rows, 0 private solution failures, visible signature/type-family/branch-loop-local/return-shape coupling
latest typed-interface private closure: public calibration skipped, private pass rate 0.083682 -> 0.39749, next-token accuracy delta 0.082302, private STS repair delta +0.05021 with 12 improvements and 0 regressions; superseded as the active private gate by the source-agnostic private closure below
latest source-agnostic private closure: public calibration skipped, private pass rate 0.062762 -> 0.485356, delta 0.422594, next-token accuracy delta 0.084484, private STS repair delta +0.05021 with 12 improvements and 0 regressions; consumed 5,722 high-transfer private rows across type/return-shape, type-contract feedback, typed-interface skeleton, admissibility/interface, edge conditions, and algorithmic planning
latest full receiver attempt after algorithmic patch: 48/128 = 0.375, quarantined because it regressed from the best 53/128 non-HumanEval diagnostic
decoder contract verifier v1: active candidate gate for signature, argument use, return shape, AST/body validity, branch/loop/local skeletons, semantic family, execution-library contracts, and STS alignment
latest private execution-shape ablation: GREEN, skeleton 64/64 = 1.0, edge-exec repair 56/64 = 0.875, semantic plan 0/64; public gate cleared with 0 no-admissible skeleton residuals and no zero-pass execution-shape categories
```

This is the current honest improvement over the older deterministic candidate
lane. Public promotion evidence must stay tied to learned token-level student
generation and must preserve anti-cheat gates: no public-solution training, no
task-id lookup, no template shortcut counted as learning, no loop-closure tool
solving benchmark tasks, and no external inference.

The active wall is not scaffolding; it is broad semantic transfer. Multi-stream
repair is positive but too small: the broad matrix shows `0.26875` aggregate
STS delta, and the latest board-run 4-card 128-task high-transfer closure shows
`0.242187` STS delta with no task-level regressions. Repeated broad runs after
fresh residual-private and high-transfer private curriculum remain below floor,
so the next real improvement must alter the decoder/architecture, not merely
regenerate similar private rows. The latest private execution-shape ablation
gate cleared and was consumed by a public receiver calibration; that run lifted
MBPP above floor but left EvalPlus, BigCodeBench, and LiveCodeBench below
floor. The source-agnostic private closure confirms the learner can absorb
type/interface/edge/algorithmic pressure privately, and STS now helps privately
without regressions. Public four-card calibration is now justified as a single
receiver measurement of that private gate, but it should not become the
debugging loop. If the receiver cards stay flat, return to decoder planning and
the remaining private zero-pass families rather than rerunning public cards.
Promotion stays blocked until a learned checkpoint clears the floor honestly
across broad clean evidence.

The latest algorithmic-planning curriculum is a useful private runway, not a
public breakthrough. It added source-agnostic private pressure for interval
merge, sliding windows, top-k frequency, graph reachability, alternating-run
state, and minimum-subarray length. The private rows are clean, and Decoder V2
now uses STS streams in skeleton admission/ranking. However, the first full
32-per-card receiver run after stratified work-budget admission regressed the
public non-HumanEval receiver score to `48/128`. That result is diagnostic-only,
not a promotion candidate. The next useful work is stronger type/interface
candidate hygiene, especially rejecting scalar-expression bodies on list,
string, and structured-interface tasks before another full public calibration.

Decoder Contract Verifier V1 now makes that candidate hygiene explicit. It
checks visible callable/signature use, return shape, syntax/body validity,
branch/loop/local skeletons, semantic family, execution-shape library contracts,
and STS alignment before candidates can support promotion. The private
execution-shape ablation report currently has 420 verifier-passing candidates
and 64 verifier failures, all tied to the intentionally failing
semantic-plan-only mode. The execution-shape skeleton decoder passed all 64
private held-out tasks, so the remaining wall is cross-card transfer rather
than the private execution-shape gate.

Rust Code LM closure reports expose `rust_work_budget_admission` so
step-budget trimming is visible. The default policy is
`legacy_sequential_work_budget_admission_v1`. A teacher-proposed stratified
admission policy balanced private high-transfer rows but regressed broad public
receiver calibration, so it is available only as an explicit diagnostic
experiment through `THESEUS_STRATIFIED_WORK_BUDGET_ADMISSION=1`. Public
receiver regression overrides private-admission aesthetics.

Current iteration-speed rule: normal Code LM work should use train-once
checkpoint fanout, not repeated training shards. The Rust fanout hot path is
split by responsibility: `candidate_fanout.rs` orchestrates report emission,
`candidate_fanout/contract_token.rs` owns contract-guided token decode and the
STS bridge, `candidate_fanout/prefilter.rs` owns verifier/ranker prefilters,
`candidate_fanout/variant_cache.rs` owns per-task parser/AST variant caching,
`fanout_timing.rs` owns candidate-task timing aggregation for reports,
`decoder_completion.rs` owns the state/SymLiquid decode loops,
`decoder_completion/parser_repair.rs` owns parser-constrained learned body
repair and AST completion variants, and `state_sequence_features.rs` owns
state-sequence static task/prompt/contract features plus the shared decoder
return-shape helper.
`task_features_io/visible_signature.rs` owns visible Python signature/import
parsing so `task_features_io.rs` stays a thinner task/feature/artifact IO
facade. `contract_verifier/quality_gate.rs` owns the cheap decoder-contract
verifier, scaffold/vacuous-body checks, beautiful-code score, and candidate
floor wall so staged verification can evolve independently from category and
semantic-contract policy. Low-latency fanout may try cheap n-gram, beam, and
greedy/readout candidates before state/SymLiquid expansion, but it must fall
through to the full families when those candidates do not satisfy the same
verifier contract. Variant-cache hits, staged fanout timings, static-feature
reuse, and module hotspot reductions are runtime-efficiency evidence, not
capability or promotion evidence.

The 2026-05-28 low-latency fanout smoke for
`code_lm_closure_speed_wall_low_latency_sts_bridge_smoke_v1` measured the
forced-token shortcut as a concrete speed win: fanout fell from `32016ms` in
the variant-cache smoke to `14734ms`, while preserving `20` private candidates,
`10` private token-level candidates, and the same public no-admissible residual.
This validates skipping feature/readout work for deterministic block tokens;
it does not unlock public calibration.

The follow-up phase-ledger smoke keeps the same candidate and residual shape
and writes `phase_timing_ms` at the fanout report top level:
`checkpoint_load_and_inputs=351ms`,
`private_candidate_generation_and_write=9188ms`,
`public_candidate_generation_and_write=4682ms`, and
`filter_diagnostics_and_gates=0ms`. The state-sequence decoder now precomputes
static task/prompt/contract features once per task for training, evaluation,
and fanout, then only rebuilds beam-context features in the hot loop. The tiny
fanout smoke stayed effectively flat (`14230ms` vs `14734ms`), so this is
training/eval cleanup and future-proofing evidence, not proof that the current
public no-admissible wall moved.

Low-latency fanout now also tries learned contract transduction before paying
the recursive STS bridge fallback. If the cheap transduction family yields a
verifier-valid candidate, the expensive bridge is skipped; otherwise the old
fallback still runs. On the same tiny smoke this preserved `20` private
candidates, `10` accepted private candidates, `10` private token-level
candidates, `1` public residual row, and the same public no-admissible reason,
with runtime `14103ms`. That small win confirms the ordering is safe, while the
remaining public `woodall_number_check` residual shows the real next capability
work is still branch/loop/local-state skeleton generation.

State-sequence and SymLiquid completion now also have a bounded in-process
memoization layer (`THESEUS_CODE_LM_DECODER_COMPLETION_CACHE`, on by default).
The cache is keyed by task/model/vocab/body-ngram/seed/limit/STS context, so
recursive bridge and ranker paths can reuse identical expensive branch
expansions without changing candidate admissibility or verifier gates. On the
tiny `speed_wall_decoder_completion_cache_smoke_v2` fanout smoke, however, the
cache preserved candidate shape but reported `0/80` hits. That makes it
larger-run reuse protection and runtime instrumentation, not a proven fix for
the current tiny public fanout wall.

No-admissible residual rows now carry `candidate_task_timing_v1` so failed
public tasks expose their branch costs instead of collapsing into an opaque
residual. The 2026-05-28 `speed_wall_no_admissible_timing_smoke_v1` row for
`source_mbpp_mbpp_20` still emitted the same public
`student_decoder_no_admissible_candidate_residual`, but it made the slow path
visible: `candidate_expression_generation=7267ms`,
`sts_conditioned_contract_token_bridge=1706ms`,
`contract_guided_token_decoder=1249ms`, `symliquid_state_bodies=1030ms`, and
`state_sequence_bodies=1113ms`. Those timings are optimizer targets only, not
capability or promotion evidence.

The next bounded speed pass split parser repair out of `decoder_completion.rs`
and ran SymLiquid/state-sequence completion families as sibling workers under
the existing parallel fanout flag. On
`speed_wall_state_symliquid_parallel_smoke_v1`, the same fixture preserved the
candidate shape (`20` private rows, `10` private token-level rows, `1` public
no-admissible residual) while fanout fell from `20445ms` to `16412ms`.
The public residual timing reported `state_symliquid_parallel_family_wall=1106ms`
against `symliquid_state_bodies=1106ms` and
`state_sequence_bodies=1043ms`, proving those two branches now overlap instead
of serializing in the low-latency path. The remaining dominant public failure
cost is still the STS bridge plus unique branch expansion, not public
calibration readiness.

The next timing cleanup moved candidate-task timing aggregation into
`fanout_timing.rs` and writes `summary.candidate_task_timing_summary` in Rust
fanout reports. A fresh 1-private/1-public probe
(`reports/code_lm_fanout_timing_probe_modular.json`) stayed GREEN at
`14420ms` and preserved a learned public token candidate, while exposing the
current branch wall directly:
`candidate_expression_generation_ms=6553ms`,
`contract_guided_token_decoder_ms=6548ms`,
`contract_pool_family_symliquid_state_ms=6518ms`, and
`contract_pool_family_state_sequence_ms=3537ms`. Those fields are optimizer
targets only. The next real speed work should reduce or cache the semantic
state/SymLiquid branch itself rather than adding speculative extra passes.

The 2026-05-29 branch optimization made two semantics-preserving decode
changes: state-sequence returns immediately when seed-prefix variants already
satisfy the requested low-latency limit, and SymLiquid readout batches two
active beams instead of waiting for four. The same 1-private/1-public probe
(`reports/code_lm_fanout_timing_probe_branch_opt.json`) stayed GREEN,
preserved the learned public candidate and verifier/guardrail pass, and moved
runtime from `14420ms` to `13739ms`. The public task timing shows the useful
part: `contract_pool_family_state_sequence_ms` fell from `3537ms` to `3ms`.
`contract_pool_family_symliquid_state_ms` remains dominant at `4551ms`, so the
next speed wall is SymLiquid branch expansion itself.

The next 2026-05-29 SymLiquid cleanup moved prompt/category/STS state features
out of the per-beam hot loop, reuses a static state base before applying
position, emitted-token, line-context, and normalization updates, and precomputes
the prompt/STS token-alignment lookup plus category/position token bonuses used
by token scoring. The same
1-private/1-public probe
(`reports/code_lm_fanout_timing_probe_token_bonus_cache.json`) stayed GREEN,
preserved the learned public candidate, and kept verifier/guardrail provenance
intact. Overall probe runtime moved from `13739ms` to `6469ms`; the public
task's `candidate_expression_generation_ms` moved from `4591ms` to `1005ms`,
and `contract_pool_family_symliquid_state_ms` moved from `4551ms` to `976ms`. The
remaining branch wall is now the per-step token option path: body n-gram lookup,
grammar/contract gating, and token scoring across the vocabulary.

The next 2026-05-29 public-metadata fanout cleanup made the low-latency path
eligible for public calibration metadata up to four candidates, but still only
under the public-boundary rule that public rows are prompt/signature metadata,
not training targets. A follow-up lazy-exit guard accepts the first
verifier-valid public metadata candidate instead of paying the full STS bridge
or exhausting all candidate slots. The canonical train-once fanout refresh
stayed GREEN, reused the checkpoint without retraining, and moved wrapper wall
time from `65337ms` to `18333ms`. Public candidate timing now reports
`candidate_expression_generation_ms=4359ms`,
`same_seed_non_sts_comparator_low_latency_ms=3356ms`,
`prompt_contract_program_decoder_fast_path_ms=638ms`, and
`rank_score_and_sort_ms=140ms`, with 32/32 public task coverage, `68`
learned token candidates, `68` verifier/guardrail passes, and `0`
template-like candidates. This is a speed/control win only; public calibration
and model growth remain locked until the private decoder gate and transfer
proof explicitly unlock them.

The next private-scale cleanup kept that public behavior intact while trimming
private-only low-latency work. The same-seed comparator now emits bounded
comparator evidence instead of re-running prototype transduction by default,
and low-latency prototype selection bounds per-key scans before expensive
verifier calls. These trims do not apply to `public_calibration` rows, because a
canonical refresh showed the richer public metadata path is needed to preserve
zero no-admissible residuals. Final verification on 2026-05-29:
`reports/code_lm_fanout_fast_path_scale_smoke64_cpt4_private_only.json` stayed
GREEN at `6041ms`, with `private_candidate_generation_and_write=2458ms`
(`38.406ms/task` wall-clock), `175` private candidates, `163` token-level
candidates, and zero public tasks. The canonical train-once fanout stayed GREEN
at `14444ms`, preserved 32/32 public task coverage, `68` verifier/guardrail
passes, `0` template-like candidates, and empty public rejection reasons. The
system-efficiency audit now treats summed internal branch timings as diagnostic
when wall-clock scale-smoke speed is already under target, so it does not chase
parallelized cumulative timing as if it were user-visible iteration time.

High-transfer private training rows are now balanced before work-budget
admission. The old capped loader was sequential, so a 2,000-row cap could load
only type/contract rows and never expose admissibility, edge, algorithmic, or
execution-shaped pressure. `scripts/code_lm_closure.py` now round-robins across
the high-transfer source files under a cap. The current private-pressure
closure loaded 5,722 high-transfer rows across type/return-shape,
type-contract feedback, typed-interface skeleton, admissibility/interface,
edge-condition, and algorithmic-planning sources before work-budget admission
trimmed the Rust training rows. It fixed the private hygiene families
`add_numbers`, `common_elements`, `median_list`, `median_odd`,
`dict_merge_three`, `title_case_words`, `safe_head`, `stable_dedupe`,
`list_chunks_every_n`, `top_k_largest`, `frequency_at_least_value`, and
`count_truthy` to `1.0`. The remaining zero-pass private families are
`list_difference`, `list_tail_replace`, `split_list_at_index`, `word_count`,
`nonempty_substring_count`, `same_chars`, and `is_anagram`.

Decoder V2 execution-shaped changes on 2026-05-20:

- public calibration sandbox creates a Windows POSIX-style `/tmp` directory so
  public tests using `/tmp/foo` do not fail before evaluating candidates;
- command-output skeletons write explicit failed-command messages and exit
  codes;
- process-control skeletons use `proc.info` or `proc.name()` and terminate all
  matching processes;
- JSON schema/email skeletons are emitted only when the visible prompt asks for
  validation/schema/email behavior.

These fixes improved a BigCodeBench 18-case diagnostic from `6/18` to `8/18`
and the latest 32-case BigCodeBench diagnostic to `8/32`. The remaining wall is
still broad algorithmic planning and receiver-card transfer, not a solved
coding learner.

## Current Conversation Learning State

Current state, 2026-05-20:

```text
conversation lane lifecycle: regression coverage unless fresh residuals appear
large benchmark: reports/high_transfer_multi_turn_conversation.json
large benchmark state: GREEN, 72 cases, 152 turns, accuracy 0.9684027777777777
hard benchmark: reports/high_transfer_multi_turn_conversation_hard.json
hard benchmark state: GREEN, 96 cases, 217 turns, accuracy 0.9772569444444442
hard benchmark graduated/saturated: true
personality-ready turns: 217/217 on the hard lane
personality runtime audit: GREEN
drift average score: 1.000
open conversation private SFT rows: 972
open conversation STS rows: 972
```

This lane is a runtime and private-training pressure surface, not public code
promotion evidence. Its purpose is to make Theseus talkable in English while
preserving memory, corrections, constraints, evidence grounding, personality
continuity, and no-leakage/public-calibration boundaries.

The 96-case hard lane previously exposed concrete session-memory residuals:
project codenames, private tokens, active work state, public-benchmark
boundary answers, and artifact-sync honesty were not always carried through.
`scripts/checkpoint_chat.py` now extracts and restates those fields explicitly,
and the follow-up hard run passed all 96 cases / 217 turns. Treat this as
operator/chat runtime evidence, not public code-generation promotion evidence.

`reports/grammar_suckers.json` now also checks the expanded conversation
pantry. The English surface grammar sucker is GREEN on 400 checked private
conversation rows with pass rate `0.8325`, and it emits SBL-lite traces for
conversation routing/STS support. The overall grammar-sucker report can remain
YELLOW when unrelated Python candidate parse residuals are present; those are
code-decoder residuals, not a conversation-lane block.

The anti-overfit rule is now explicit: do not train private rows merely because
a benchmark name is weak. Public failures may define source-agnostic residual
concepts, but private pressure should vary signatures, return shapes,
edge cases, and interfaces so a fix on one card can transfer to receiver cards.
`reports/transfer_generalization_audit.json` is the guardrail for this.
`scripts/code_residual_curriculum.py` is now the residual-summary and report
writer, while private generated task templates and variant rendering live in
`scripts/code_residual_curriculum_templates.py`. This keeps the curriculum
runner small enough for AI maintenance without changing the private-row schema
or the rule that public prompts/tests/solutions remain excluded from training.
`scripts/real_code_benchmark_graduation.py` is now the thin graduation
orchestrator; constants, dataset loaders, staged verification/runtime, and
support/report helpers live in `scripts/real_code_benchmark_constants.py`,
`scripts/real_code_benchmark_datasets.py`,
`scripts/real_code_benchmark_runtime.py`, and
`scripts/real_code_benchmark_support.py`. The public function surface is kept
for downstream importers while benchmark speed work can target verification or
dataset loading without touching report governance.

Use these reports before quoting any score:

- `reports/learning_scoreboard.json`
- `reports/real_code_benchmark_graduation.json`
- `reports/candidate_promotion_gate.json`
- `reports/code_lm_closure.json`
- `reports/code_lm_closure_rust.json`
- `reports/sts_native_parallel_probe.json`
- `reports/benchmaxx_curriculum.json`
- `reports/training_resource_runway.json`

`reports/training_resource_runway.json` is the current source-of-truth summary
for what training and benchmark resources are staged. It separates private
training pressure from public calibration assets. As of 2026-05-18 it reports a
green runway with 50 source repos on `D:\ProjectTheseus`, BigCodeBench/EvalPlus/
LiveCodeBench public calibration assets staged on D:, 755 counted public
calibration tasks across HumanEval/MBPP/EvalPlus, 92 benchmark adapter cards,
55 smoke-passed adapter cards, 58 local RL environments, 7,081 private training
rows, 913 STS rows, and 608 governed small-sample rows. Public benchmark assets
remain scorer/calibration-only and must not enter private training.

## Public Benchmark Source Fetching

Fetch public benchmark definitions and local environment references with:

```bash
python scripts/fetch_public_benchmarks.py --out-root data/public_benchmarks
```

This clones public repos such as BLIMP, the BabyLM evaluation pipeline, and
PufferLib/Ocean references, then writes a manifest with source URLs and pinned
commits. The fetcher downloads benchmark assets only; it does not call hosted
model APIs or run external inference.

Convert the fetched public BLIMP files into balanced local splits:

```bash
python scripts/export_public_blimp.py --data-dir data/public_benchmarks/blimp/data --out-train data/public_blimp_train.jsonl --out-eval data/public_blimp_eval.jsonl --eval-fraction 0.1 --seed 0
```

## Benchmark Adapter Smoke Sweep

The adapter factory turns governed source cards, local ROM profiles, and
resource-pantry clones into explicit benchmark cards under `benchmarks/cards/`.
Before a card becomes frontier pressure, run the local-only smoke sweep:

```bash
.\.venv-puffer\Scripts\python.exe scripts\benchmark_adapter_smoke.py --out reports\benchmark_adapter_smoke_status.json --markdown-out reports\benchmark_adapter_smoke_status.md
python scripts\benchmark_adapter_factory.py --write-cards --out reports\benchmark_adapter_factory.json --markdown-out reports\benchmark_adapter_factory.md
python scripts\rl_benchmark_registry.py --refresh-local --out reports\rl_benchmark_registry.json
python scripts\benchmaxx_curriculum.py --out reports\benchmaxx_curriculum.json --markdown-out reports\benchmaxx_curriculum.md
```

`scripts/benchmaxx_curriculum.py` stays the CLI and curriculum policy owner.
Rendering, JSON/text IO, path lookup, numeric coercion, and timestamp helpers
live in `scripts/benchmaxx_curriculum_io.py`; this keeps the benchmark
curriculum plane below the Python soft maintainability limit without changing
the report contract consumed by autonomy, dashboard, and calibration schedulers.

`scripts/pressure_runner.py` stays the local pressure-card orchestration layer.
Generic path/JSON/budget/timeout helpers live in `scripts/pressure_runner_utils.py`,
and code-benchmark transfer evidence scoring lives in
`scripts/pressure_runner_code_evidence.py`; this keeps benchmark pressure
dispatch below the Python soft maintainability limit while preserving the
pressure report schema and calibration-only public benchmark rules.

Last recorded full adapter sweep, 2026-05-13:

```text
cards smoke-classified: 92 / 92
adapter smoke passed: 55
runtime blocked: 13
governance or asset blocked: 24
failed: 0
external inference calls: 0
candidate flow ready: true
```

The smoke-passed RL queue currently includes Brax, bsuite, Craftax, Crafter,
DeepMind Control, Gymnasium, Jumanji, Minigrid, EnvPool, Meta-World, and
PettingZoo. The drone queue now also includes smoke-passed MAVSDK-Python,
PyFlyt, and gym-pybullet-drones lanes. The emulator stage has user ROM metadata,
but the GBA wrapper is runtime-blocked until native mGBA Python bindings are
installed/built. Procgen is staged but blocked for Windows native Qt/CMake build
requirements.

The coding pressure lane now includes local code and coding-agent surfaces:
HumanEval, MBPP, EvalPlus, BigCodeBench, LiveCodeBench, SWE-bench,
mini-SWE-agent, SWE-agent, OpenCode, OpenHands, Terminal-Bench, CodeClash,
SWE-Atlas, SWE-PolyBench, SWE-ReX, SWE-smith, and SWE-gen. OpenCode Bench,
RepoTransBench, RepoBench, SWELancer, The Stack v2, StarCoderData,
CodeSearchNet, and other unclear or queue-only sources are excluded from active
benchmark and training rotation until permissive license/terms are explicit.
Coding-agent cards are
scored only through local Theseus/OpenAI-compatible endpoints; provider API-key
paths are audited as forbidden during scoring and do not count as system
capability.

Current coding pressure smoke status:

```text
EvalPlus: passed
SWE-bench: passed as coding_agent_harness_adapter
mini-SWE-agent: passed
SWE-agent: passed
OpenCode: passed; Bun is available through ignored project-local toolchain storage
CodeClash: passed
SWE-PolyBench: passed
SWE-gen: passed
APPS metadata card: passed
CodeContests metadata card: passed
OpenHands: metadata-passed, runtime-blocked on Docker or Podman for full sandbox execution
Terminal-Bench: metadata-passed, runtime-blocked on Docker or Podman for full sandbox execution
SWE-ReX: metadata-passed, runtime-blocked on Docker or Podman for full sandbox execution
SWE-smith: metadata-passed, runtime-blocked on Docker or Podman for full sandbox execution
SWE-Atlas: staged, blocked until a manifest/loader and Docker or Podman runtime are present
OpenCode Bench and all unclear/queue-only sources: excluded from active runway until permissive license/terms are explicit
```

Refresh local coding runtime readiness with:

```powershell
.\scripts\setup_coding_runtime.ps1
.\scripts\setup_coding_runtime.ps1 -InstallBun
.\scripts\setup_coding_runtime.ps1 -InstallPodman
.\scripts\setup_podman_sandbox_windows.ps1
.\scripts\setup_coding_runtime.ps1 -Smoke
```

The Bun remediation is intentionally local to ignored storage:
`data/external_benchmark_candidates/toolchains/bun/`. Docker Desktop or Podman
is still a host runtime decision; the smoke and pressure runners accept either
Docker or Podman for full container-backed coding harnesses. Without a
container runtime, those cards still produce source-contract pressure and
explicit `container_runtime_missing` residuals instead of silently calling
external services.

Candidate bottlenecks are reduced before teacher escalation with:

```powershell
python scripts\candidate_bottleneck_reducer.py --fix --out reports\candidate_bottleneck_reducer.json
```

The reducer may create isolated local runtimes and refresh reports, but it does
not call external inference, bulk-download data, or start live hardware. Current
state is `YELLOW_OPTIONAL_RUNTIME_BLOCKERS`: the next frontier is runnable, no
safe auto-remediation remains, and the remaining runtime blockers are explicit
manual/native setup lanes rather than teacher work.

## Board Game RL: Chess And Go

Chess and Go are now first-class local RL/self-play lanes for long-horizon
planning, legal-action masking, sparse reward, tactical diagnostics, and
Elo-style improvement tracking. The runner is:

```powershell
python scripts\board_game_rl_benchmark.py --games chess,go --seed 14 --chess-games 24 --go-games 24 --go-board-size 5 --out reports\board_game_rl_benchmark.json --markdown-out reports\board_game_rl_benchmark.md --ratings-out reports\board_game_elo_ratings.json --trace-out reports\board_game_rl_traces.jsonl
python scripts\rl_benchmark_registry.py --refresh-local --out reports\rl_benchmark_registry.json
```

Claim-bearing outputs:

```text
reports/board_game_rl_benchmark.json
reports/board_game_rl_benchmark.md
reports/board_game_elo_ratings.json
reports/board_game_elo_history.jsonl
reports/board_game_rl_traces.jsonl
benchmarks/cards/source_chess_rl.json
benchmarks/cards/source_go_rl.json
```

The chess lane uses `python-chess` for legal move generation and rules. The Go
lane uses a project-local small-board rules engine with captures, suicide
prevention, simple ko, pass moves, and area scoring. Both lanes start with
bounded policy self-play and tactical diagnostics rather than external engines.

Policy:

```text
external inference calls: 0
public benchmark answers: forbidden as training data
engine/pro game corpora: require separate license and contamination review
Elo deltas: trend evidence only, not single-run promotion proof
graduation: stable tactical gates become regression before board/opponent scale-up
```

The high-transfer scheduler exposes this as the medium-priority
`board_game_rl` concept so unattended runs can use idle slots for strategic
games without displacing the current code-transfer wall.

## Drone Racing / AI Grand Prix Lane

The drone lane is now a first-class governed benchmark family. It is simulation
only by default, live hardware is forbidden without explicit human approval,
and external inference is forbidden inside the control loop.

Last recorded drone-lane snapshot, 2026-05-13:

```text
AI Grand Prix spec digest: reports/ai_grand_prix_spec_digest.json
known-good competition Python: 3.14.2
current Python 3.14 lane: missing, reports/python_runtime_compatibility.json is YELLOW_RUNTIME_ACTION_REQUIRED
drone source candidates: 7
drone adapter cards: 7
drone adapter smoke-passed: 3
drone runtime-blocked: 4
next runnable frontier: source_gym_pybullet_drones
```

Staged or tracked sources include PyFlyt, gym-pybullet-drones, AirSim Drone
Racing Lab, AirSim, MAVSDK-Python, PX4 SITL, and Aerial Gym Simulator. Their
cards are written under `benchmarks/cards/` and are routed by the Octopus
Router through `drone_racing_control_arm`, `python_runtime_compliance_arm`, and
`safety_reflex_arm`.

For safe local development lanes:

```powershell
.\scripts\setup_drone_runtime.ps1 -Lane pyflyt
.\scripts\setup_drone_runtime.ps1 -Lane gym-pybullet
.\scripts\setup_drone_runtime.ps1 -Lane control-client
```

Those commands create isolated Python 3.11 venvs for simulator/control smoke.
For the official AI Grand Prix lane:

```powershell
.\scripts\setup_drone_runtime.ps1 -Lane competition
```

That command expects Python 3.14 through the Windows `py` launcher and creates
`.venv-drone-py314`. For local competition-shape development before Python 3.14
is installed, use:

```powershell
.\scripts\setup_drone_runtime.ps1 -Lane competition -UsePython311DevFallback
```

The development fallback is not the official AI Grand Prix lane. Before an
official race submission, the runtime report must show
`ai_grand_prix_runtime_ready=true`, the simulator endpoint smoke must pass, and
the control adapter must record command-rate, heartbeat, vision packet, and
no-human-interaction compliance.

## Native Voice STT/TTS Data Lane

Voice remains a native Theseus head/router capability. The project must not
install or call borrowed STT/TTS inference stacks to satisfy voice gates.
Licensed speech corpora are used only as pressure and training data for local
Theseus learners.

Current governed sources:

| Source | Role | License Mode | Default Action |
| --- | --- | --- | --- |
| LibriSpeech | STT train/eval, audio-text alignment, TTS bootstrap | CC BY 4.0 | Tiny governed validation-clean shards may be materialized automatically. |
| LibriTTS | TTS train/eval | CC BY 4.0 | Metadata-only until a shard downloader and storage budget are approved. |
| LJSpeech | Single-speaker TTS train/eval | Public domain | Metadata-only until a tiny TTS shard fetcher is approved. |
| Common Voice | Multilingual STT train/eval | CC0-1.0 | Metadata-only until locale/privacy/storage policy selects a shard. |
| VCTK | Speaker-diverse TTS eval/train | ODC-By | Queue-only until attribution/storage review is complete. |

Refresh the native voice manifest with:

```powershell
python scripts\native_voice_training_manifest.py --allow-network-fetch --out reports\native_voice_training_manifest.json
python scripts\native_voice_bootstrap_learner.py --manifest reports\native_voice_training_manifest.json --stt-out reports\native_stt_decoder.json --tts-out reports\native_tts_generator.json
python scripts\native_voice_io.py --out reports\native_voice_io.json
```

The manifest writes ignored tiny audio/transcript artifacts under
`data/external_benchmark_candidates/native_voice_samples/` and emits training
packets for `native_stt_decoder` and `native_tts_generator`. Those component
reports must come from local Theseus training, not Whisper, Vosk, SpeechBrain
pretrained models, pyttsx3, cloud APIs, or other external speech inference.
The bootstrap learner is only a local index/plumbing proof and explicitly keeps
`native_model_ready=false` until real native STT/TTS training reports metrics.

## Current Local Training Command

```bash
cargo run --release -p symliquid-cli -- train-standalone --train-seed 0 --eval-seed 10000 --cases-per-task 500 --epochs 30 --batch-size 1 --hv-dim 4096 --model-out reports/symliquid_policy_500x30_hv4096.json --out reports/symliquid_cached_sgd_500x30_bs1_lr005_runtime.json
```

CUDA readout training is available when the CLI is built with `--features cuda`:

```bash
cargo run --release -p symliquid-cli --features cuda -- train-standalone-cuda --train-seed 0 --eval-seed 10000 --cases-per-task 100 --epochs 10 --samples-per-launch 32 --hv-dim 4096 --lr 0.05 --out reports/symliquid_cuda_sgd_100x10_hv4096.json
```

CUDA rollout-backed training evolves liquid/reservoir/VSA state on GPU before
training the local readout:

```bash
cargo run --release -p symliquid-cli --features cuda -- train-rollout-cuda --train-seed 0 --eval-seed 10000 --cases-per-task 50 --epochs 5 --state-epochs 6 --state-lr 0.02 --samples-per-launch 32 --rollout-batch 200 --obs-dim 64 --hidden-dim 96 --reservoir-dim 128 --hv-dim 1024 --seq-len 64 --lr 0.03 --out reports/symliquid_rollout_cuda_50x5_hv1024_full_governance_lr002_e6.json
```

The rollout trainer now probes three readout routes on independent synthetic
cases:

```text
shared CUDA rollout readout
  + low-rank task-conditioned residual adapters
  + full task heads as a guarded diagnostic backup
```

Residual adapters are promoted only when the probe improves aggregate accuracy
by at least `+0.02` and the worst per-family task delta stays above `-0.05`.
This targets specialization without letting a task-specific head damage action,
belief, or verifier-governed families.

Use a sweep when tuning state updates. The sweep records every run, tracks the
best held-out result, and counts how often the guarded state-update candidate is
accepted:

```bash
cargo run --release -p symliquid-cli --features cuda -- train-rollout-cuda-sweep --train-seeds 0,1,2 --eval-seed-base 10000 --cases-per-task 50 --epochs 5 --state-epochs 0,2,6 --state-lrs 0.0,0.005,0.02 --samples-per-launch 32 --rollout-batch 200 --obs-dim 64 --hidden-dim 96 --reservoir-dim 128 --hv-dim 1024 --seq-len 64 --lr 0.03 --out reports/symliquid_rollout_cuda_sweep.json
```

The current standalone learned-transfer model is intentionally simple:

```text
observation text
  -> deterministic hash/VSA-style hypervector features
  -> structured CGS/VSA answer, evidence, pairwise, and verifier features
  -> local trained readout
  -> exact verifier scoring
```

Symbolic fallback is off by default for this path. This is the first serious
local transfer benchmark for SymLiquid, not the final model.

## Benchmark Families

Generated hard suites include:

| Family | Purpose |
| --- | --- |
| `role_filler` | symbolic binding and cleanup |
| `long_context_role_filler` | memory under distractor load |
| `active_classification` | epistemic inspection before classification |
| `gridworld` | belief/action governance under hidden state |
| `missing_evidence_rag` | inspect/abstain when evidence is absent |
| `code_repair_verifier` | select patch that satisfies a verifier |
| `babylm_minimal_pair` | BabyLM-style grammatical minimal-pair preference |
| `blimp_acceptability` | BLIMP-like acceptability under attractors/dependencies |
| `long_context_retrieval` | retrieve verified values from noisy context |
| `adversarial_rag` | resist unsupported or conflicting evidence |

Generate a frozen suite:

```bash
cargo run -p symliquid-cli -- benchmark-snapshot --seed 0 --cases-per-task 20 --out benchmarks/snapshots/cgs_hard_seed0.json
```

Run the reference verifier sanity path:

```bash
cargo run -p symliquid-cli -- benchmark-symliquid --suite benchmarks/snapshots/cgs_hard_seed0.json --out reports/symliquid_reference_report.json
```

The `benchmark-symliquid` reference sanity path derives answers from the
observation through local symbolic/VSA-style parsing and governance rules. The
`train-standalone` learned-transfer path disables symbolic fallback by default
and scores the local trained readout. Neither path calls external models or
reads the expected output as a shortcut.

Latest local measured run:

```text
Command: cargo run --release -p symliquid-cli -- train-standalone --train-seed 0 --eval-seed 10000 --cases-per-task 500 --epochs 30 --batch-size 1 --hv-dim 4096 --model-out reports/symliquid_policy_500x30_hv4096.json --out reports/symliquid_cached_sgd_500x30_bs1_lr005_runtime.json
Cases: 5000 held-out generated cases
Accuracy: 0.997
Residual: 0.003
Invalid action rate: 0.000
External inference calls: 0
Train examples/sec: 2227.8
Local text-only baseline: accuracy=0.774 residual=0.226
Lift: +0.222 accuracy, -0.222 residual
Report: reports/symliquid_cached_sgd_500x30_bs1_lr005_runtime.json
Comparison: reports/compare_text_baseline_vs_symliquid_cached_sgd_500x30_runtime.json
Policy artifact: reports/symliquid_policy_500x30_hv4096.json
```

Latest CUDA readout smoke:

```text
Command: cargo run --release -p symliquid-cli --features cuda -- train-standalone-cuda --train-seed 0 --eval-seed 10000 --cases-per-task 100 --epochs 10 --samples-per-launch 32 --hv-dim 4096 --lr 0.05 --out reports/symliquid_cuda_sgd_100x10_hv4096.json
Cases: 1000 held-out generated cases
Accuracy: 0.787
Residual: 0.213
Invalid action rate: 0.000
External inference calls: 0
Train examples/sec: 2312.7
CPU same config: accuracy=0.787 residual=0.213 train_examples/sec=1981.4
Local text-only baseline: accuracy=0.631 residual=0.369
Lift over text-only baseline: +0.156 accuracy, -0.156 residual
Report: reports/symliquid_cuda_sgd_100x10_hv4096.json
Comparison: reports/compare_baseline_vs_symliquid_cuda_100x10_hv4096.json
```

Latest CUDA rollout smoke:

```text
Command: cargo run --release -p symliquid-cli --features cuda -- train-rollout-cuda --train-seed 0 --eval-seed 10000 --cases-per-task 50 --epochs 5 --state-epochs 6 --state-lr 0.02 --samples-per-launch 32 --rollout-batch 200 --obs-dim 64 --hidden-dim 96 --reservoir-dim 128 --hv-dim 1024 --seq-len 64 --lr 0.03 --out reports/symliquid_rollout_cuda_50x5_hv1024_full_governance_lr002_e6.json
Cases: 500 held-out generated cases
Accuracy: 1.000
Residual: 0.000
Invalid action rate: 0.000
Train examples/sec: 288.1
Feature set: cuda_liquid_reservoir_vsa_rollout_entity_slot_memory_trainable_state_retrieval_curriculum_shared_head
Readout router: selected shared_head
Readout probe: shared=1.000 residual_adapters=1.000 rank=60 worst_residual_delta=+0.000 task_heads=1.000 worst_task_delta=+0.000
CGS governance routing: entity-slot memory, active information seeking, verified-evidence RAG, and pairwise grammar priors
State update candidate: accepted
Probe cases: 1000 independent synthetic cases
Probe metric: masked allowed-action or answer-candidate accuracy
Base probe: accuracy=0.943 loss=9.2104 alignment=0.0026
Candidate probe: accuracy=1.000 loss=7.9394 alignment=0.2797
Worst task accuracy delta: +0.000
Task-gated candidate: rejected
Candidate state tasks: gridworld
Local text-only baseline: accuracy=0.504 residual=0.496
Lift over text-only baseline: +0.496 accuracy, -0.496 residual
Three-seed sweep: mean accuracy=1.000 std=0.000 mean residual=0.000
Report: reports/symliquid_rollout_cuda_50x5_hv1024_full_governance_lr002_e6.json
Comparison: reports/compare_baseline_vs_symliquid_rollout_cuda_50x5_hv1024_full_governance.json
Sweep: reports/symliquid_rollout_cuda_sweep_3seed_50x5_hv1024_full_governance.json
```

This generated suite is now best treated as a local contract/verifier suite,
not as an open-ended learning benchmark. The next pressure test should move the
same governed-memory architecture onto public BLIMP/BabyLM splits, adversarial
RAG variants not generated by these templates, and Puffer/Ocean rollouts.

## Local Baselines To Add

The fair baselines all run locally:

| Baseline | Why |
| --- | --- |
| `first_allowed` | trivial action floor |
| `bag_of_words` | weak lexical/rule floor |
| `hash_readout` | text-only trained hash readout without structured CGS/VSA features |
| reservoir-only | planned future ablation isolating recurrent expansion |
| VSA-only | planned future ablation isolating symbolic binding |
| liquid/reservoir/VSA model | full local SymLiquid candidate |
| tiny local RNN/GRU | conventional recurrent control if implemented |

Commands:

```bash
cargo run -p symliquid-cli -- benchmark-baseline --suite benchmarks/snapshots/cgs_hard_seed0.json --baseline bag_of_words --out reports/local_bow_baseline_report.json
cargo run -p symliquid-cli -- benchmark-baseline --suite benchmarks/snapshots/cgs_hard_seed0.json --baseline hash_readout --epochs 10 --hv-dim 2048 --out reports/local_hash_readout_baseline_report.json
cargo run --release -p symliquid-cli -- train-baseline --train-seed 0 --eval-seed 10000 --cases-per-task 500 --epochs 30 --hv-dim 4096 --out reports/local_text_hash_transfer_seed0_eval10000_500x30_hv4096.json
cargo run --release -p symliquid-cli -- benchmark-compare --baseline reports/local_text_hash_transfer_seed0_eval10000_500x30_hv4096.json --candidate reports/symliquid_cached_sgd_500x30_bs1_lr005_runtime.json --out reports/compare_text_baseline_vs_symliquid_cached_sgd_500x30_runtime.json
cargo run --release -p symliquid-cli -- benchmark-compare --baseline reports/local_text_hash_cpu_100x10_hv4096.json --candidate reports/symliquid_cuda_sgd_100x10_hv4096.json --out reports/compare_baseline_vs_symliquid_cuda_100x10_hv4096.json
cargo run --release -p symliquid-cli -- benchmark-compare --baseline reports/local_text_hash_cpu_50x5_hv1024.json --candidate reports/symliquid_rollout_cuda_50x5_hv1024_readout_gated_lr002_e6.json --out reports/compare_baseline_vs_symliquid_rollout_cuda_50x5_hv1024_readout_gated.json
```

Use `--batch-size 1` for the current quality-preserving run. Larger batches are
available for accelerator experiments, but they need optimizer tuning to match
the long-context cleanup accuracy.

Seed sweep:

```bash
cargo run -p symliquid-cli -- seed-sweep --train-seeds 0,1,2 --eval-seed-base 10000 --cases-per-task 20 --epochs 10 --hv-dim 2048 --out reports/symliquid_seed_sweep.json
```

BabyLM/BLIMP local probe bridge:

```bash
cargo run -p symliquid-cli -- babylm-probe --input "C:\Users\corbe\Documents\babylm-candidate\data\samples\strict_small_50k_words.txt" --seed 0 --limit 50 --out-suite benchmarks/snapshots/babylm_local_probe_smoke.json --out-report reports/babylm_local_probe_smoke.json
```

Extended BabyLM probe training:

```bash
cargo run -p symliquid-cli -- train-babylm-probe --input "C:\Users\corbe\Documents\babylm-candidate\data\samples\strict_small_500k_words.txt" --train-seed 0 --eval-seed 10000 --train-limit 5000 --eval-limit 1000 --steps 5000 --hv-dim 8192 --lr 0.05 --out reports/babylm_probe_train_5k.json
```

Export and run the local cached BabyLM BLIMP filtered split:

```bash
python scripts/export_blimp_filtered.py --cache-root "C:\Users\corbe\Documents\babylm-candidate\.cache\huggingface\datasets\BabyLM-community___baby_lm-blimp-filtered" --out-train data/babylm_blimp_filtered_train.jsonl --out-eval data/babylm_blimp_filtered_eval.jsonl --eval-fraction 0.1 --seed 0
cargo run -p symliquid-cli -- train-babylm-probe --input data/babylm_blimp_filtered_train.jsonl --eval-input data/babylm_blimp_filtered_eval.jsonl --train-limit 53888 --eval-limit 5987 --steps 800000 --hv-dim 16384 --lr 0.2 --out reports/blimp_filtered_train_800k_evalfull_hv16k_lr02_complexnpfix.json
```

Run a public BLIMP smoke on the fetched official split:

```bash
cargo run --release -p symliquid-cli -- train-babylm-probe --input data/public_blimp_train.jsonl --eval-input data/public_blimp_eval.jsonl --train-limit 10000 --eval-limit 1000 --steps 20000 --hv-dim 8192 --lr 0.05 --pairwise-contrast --balance-rules --out reports/public_blimp_probe_train10k_eval1k_steps20k_hv8k.json
```

Latest public BLIMP pressure test:

```text
Architecture delta: BLIMP linguistic state v3
  - c-command/reconstruction reflexive state
  - passive/animacy/argument-structure state
  - existential-there and expletive-it governance state
  - adjunct/left-branch island cues
  - direct pairwise linguistic governance delta

Command: cargo run --release -p symliquid-cli -- train-babylm-probe --input data/public_blimp_train.jsonl --eval-input data/public_blimp_eval.jsonl --train-limit 30000 --eval-limit 6700 --steps 160000 --hv-dim 8192 --lr 0.05 --pairwise-contrast --balance-rules --prior-weight 0.25 --out reports/public_blimp_probe_train30k_eval6700_steps160k_hv8k_state_v3_prior025.json
Public train pairs: 30000
Public holdout pairs: 6700
Accuracy: 0.826
Residual: 0.174
External inference calls: 0
Report: reports/public_blimp_probe_train30k_eval6700_steps160k_hv8k_state_v3_prior025.json

Three-seed sweep:
mean accuracy=0.82498
std accuracy=0.00062
mean residual=0.17502
std residual=0.00062
Sweep: reports/public_blimp_probe_train30k_eval6700_steps160k_hv8k_state_v3_prior025_seed_sweep.json
```

Unseen adversarial RAG variants:

```bash
python scripts/generate_unseen_adversarial_rag.py --count 180 --seed 17 --out benchmarks/snapshots/unseen_adversarial_rag_seed17.json
cargo run --release -p symliquid-cli -- benchmark-symliquid --suite benchmarks/snapshots/unseen_adversarial_rag_seed17.json --model-id symliquid-unseen-rag-governance --out reports/symliquid_unseen_adversarial_rag_seed17.json
cargo run --release -p symliquid-cli -- benchmark-baseline --suite benchmarks/snapshots/unseen_adversarial_rag_seed17.json --baseline bag_of_words --out reports/baseline_bag_of_words_unseen_adversarial_rag_seed17.json
cargo run --release -p symliquid-cli -- benchmark-compare --baseline reports/baseline_bag_of_words_unseen_adversarial_rag_seed17.json --candidate reports/symliquid_unseen_adversarial_rag_seed17.json --out reports/compare_unseen_adversarial_rag_seed17_bow_vs_symliquid.json
```

```text
Cases: 180
SymLiquid accuracy: 1.000
Bag-of-words baseline accuracy: 0.333
Accuracy lift: +0.667
Residual delta: -0.667
External inference calls: 0
Report: reports/symliquid_unseen_adversarial_rag_seed17.json
Comparison: reports/compare_unseen_adversarial_rag_seed17_bow_vs_symliquid.json
```

Latest public BLIMP smoke:

```text
Official BLIMP-derived local split: 60,300 train / 6,700 eval across 67 rules
Train pairs: 10,000
Eval pairs: 1,000
Steps: 20,000
HV dim: 8,192
Accuracy: 0.708
Residual: 0.292
External inference calls: 0
Report: reports/public_blimp_probe_train10k_eval1k_steps20k_hv8k.json
```

The BabyLM bridge accepts local text, JSONL, CSV, or TSV minimal-pair sources.
It never calls a model provider. Text input is converted into local
corpus-corruption pairs, so it should be treated as a diagnostic probe rather
than an official BabyLM score. JSONL splits can be kept disjoint with
`--eval-input`.

Latest smoke measurements:

```text
cgs_hard_governance learned transfer, 5000 held-out generated cases:
  local text-only baseline:       accuracy=0.774 residual=0.226
  SymLiquid structured CGS/VSA:   accuracy=0.997 residual=0.003
  lift:                           +0.222 accuracy, -0.222 residual
  remaining residual:             long_context_role_filler=0.034, long_context_retrieval=0.000

cuda_readout_sgd, 1000 held-out generated cases:
  local text-only baseline:       accuracy=0.631 residual=0.369
  SymLiquid CPU readout:          accuracy=0.787 residual=0.213 train_examples/sec=1981.4
  SymLiquid CUDA readout:         accuracy=0.787 residual=0.213 train_examples/sec=2312.7
  lift vs text-only baseline:     +0.156 accuracy, -0.156 residual

cuda_rollout_state, 500 held-out generated cases:
  local text-only baseline:       accuracy=0.504 residual=0.496
  SymLiquid CUDA rollout:         accuracy=0.582 residual=0.418 train_examples/sec=131.9
  readout router:                 selected shared_head
  readout probe:                  shared=0.602 task_heads=0.523 worst_task_delta=-0.570
  state update candidate:         rejected by probe gate
  probe metric:                   masked accuracy on 1000 independent synthetic cases
  probe gate:                     base=0.609 candidate=0.592 worst_task_delta=-0.350
  task-gated candidate:           rejected; probe-positive task route did not pass full gate
  lift vs text-only baseline:     +0.078 accuracy, -0.078 residual

cgs_hard_smoke, 100 cases:
  SymLiquid symbolic governance: accuracy=1.000 residual=0.000
  local bag_of_words baseline:   accuracy=0.650 residual=0.350
  local hash_readout baseline:   accuracy=0.480 residual=0.520

babylm_local_probe_smoke, 50 corpus-corruption cases:
  SymLiquid current scorer:      accuracy=0.400 residual=0.600

babylm_local_probe, 1000 held-out cases from 500k-word local sample:
  first_allowed baseline:        accuracy=0.492 residual=0.508
  bag_of_words baseline:         accuracy=0.492 residual=0.508
  hash_readout baseline:         accuracy=0.593 residual=0.407
  SymLiquid sequence scorer 1k:  accuracy=0.816 residual=0.184
  SymLiquid sequence scorer 5k:  accuracy=0.814 residual=0.186
  SymLiquid sequence scorer 50k: accuracy=0.854 residual=0.146

local BabyLM BLIMP filtered split, 53,888 train / 5,987 eval:
  contract symbolic scorer:      accuracy=0.500 residual=0.500
  SymLiquid 50k steps:           accuracy=0.746 residual=0.254
  SymLiquid 100k steps + state:  accuracy=0.810 residual=0.190
  SymLiquid 400k + binding/island: accuracy=0.890 residual=0.110
  SymLiquid 800k + binding/island/SVA: accuracy=0.924 residual=0.076

seed_sweep_smoke, seeds 0 and 1:
  mean_accuracy=1.000 std_accuracy=0.000
```

The early 1k-to-5k plateau was useful feedback: the lexical pairwise scorer
learned the easy corruption families quickly, then saturated. The BLIMP split
improved only after adding trainable linguistic state features for
determiner-noun binding, reflexive binding, filler-gap/island structure,
tough/raising adjective classes, ellipsis state, and subject-verb head binding.

## Reporting

Every run should report:

```text
task
variant
train_seed
eval_seed
train_examples
eval_examples
accuracy
residual
invalid_action_rate
runtime_ms
memory_bytes
parameter_count
verified_success_per_cost
```

## Public Score Comparison

Use `configs/public_baselines.toml` to record published scores from other
systems. A public score is comparable only when SymLiquid runs the same
benchmark, split, metric, and data budget.

## Local Competition Artifacts Found

Existing related work on this machine:

```text
C:\Users\corbe\Documents\babylm-candidate
C:\Users\corbe\Documents\golf
```

The next BabyLM/Parameter Golf integration should read benchmark contracts,
data manifests, and existing scripts from those folders, but keep SymLiquid's
training and inference local to this repository unless we intentionally export a
candidate artifact.

## PufferLib / RL Environments

PufferLib should be treated as a local environment and rollout harness, not as a
model provider. The adapter surface lives in:

```text
adapters/pufferlib/
```

Smoke check:

```bash
python adapters/pufferlib/symliquid_puffer_adapter.py --check
```

Local setup completed on this machine:

```text
clone: vendor/pufferlib
commit: 69fcbcff
venv: .venv-puffer
python: 3.11.9
pufferlib: 4.0 editable install
torch: 2.11.0+cpu
compiled pufferlib._C backend: unavailable on this Windows editable install
```

Use the dedicated venv:

```bash
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --check
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --artifact reports\symliquid_policy_500x30_hv4096.json --smoke-actions 8
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --artifact reports\symliquid_policy_500x30_hv4096.json --rollout-smoke-steps 128 --num-envs 64 --obs-dim 8 --action-modulo 4 --out reports/puffer_style_smoke_128x64.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --artifact reports\symliquid_policy_500x30_hv4096.json --env ocean-cartpole --rollout-smoke-steps 512 --num-envs 64 --out reports/puffer_ocean_cartpole_smoke_512x64.json
```

The current adapter can load a local SymLiquid readout artifact emitted by
`--model-out`, score sparse hashed policy features, and run local vectorized
Ocean-style control loops ported from the vendored PufferLib Ocean dynamics.
The high-throughput target is to keep Python at the environment boundary while
moving policy rollout and state updates into Rust/CUDA. The current CUDA
readout trainer already uses persistent feature/target/weight buffers, and
`rollout_state_update_kernel` now parity-tests a batched liquid/reservoir/VSA
state update against the CPU implementation. The Rust FFI bridge now moves
dense, cue-memory, T-maze, and noisy-evidence recurrent policy scoring out of
Python. It also owns the local discrete CEM rollout/training loop for chain,
memory, T-maze, noisy-memory, noisy-T-maze, and slot-T-maze tasks. The next bottleneck is moving the same
step/reward/done/optimizer state into CUDA-backed batched kernels.

Latest local Puffer-style adapter smoke:

```text
Command: .\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --artifact reports\symliquid_policy_500x30_hv4096.json --rollout-smoke-steps 1024 --num-envs 128 --obs-dim 8 --action-modulo 4 --out reports/puffer_style_smoke_1024x128_state_v3.json
Transitions: 131072
Transitions/sec: 10719.2
Mean reward: 0.250
External inference calls: 0
Report: reports/puffer_style_smoke_1024x128_state_v3.json
Compiled pufferlib._C backend: unavailable on this Windows editable install
```

Latest local Ocean CartPole-style adapter smoke:

```text
Command: .\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --artifact reports\symliquid_policy_500x30_hv4096.json --env ocean-cartpole --rollout-smoke-steps 1024 --num-envs 128 --out reports/puffer_ocean_cartpole_smoke_1024x128_state_v3.json
Transitions: 131072
Transitions/sec: 18275.7
Mean reward: 0.816
External inference calls: 0
Report: reports/puffer_ocean_cartpole_smoke_1024x128_state_v3.json
```

Latest Ocean CartPole-style smoke with explicit reflex-governance overlay:

```text
Command: .\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --artifact reports\symliquid_policy_500x30_hv4096.json --env ocean-cartpole --rollout-smoke-steps 1024 --num-envs 128 --governance-reflex --out reports/puffer_ocean_cartpole_smoke_1024x128_state_v3_reflex.json
Transitions: 131072
Transitions/sec: 18534.3
Mean reward: 0.995
Dones: 0
Truncations: 640
Reflex overrides: 65530
External inference calls: 0
Report: reports/puffer_ocean_cartpole_smoke_1024x128_state_v3_reflex.json
```

Learned local Ocean CartPole policy:

```bash
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --train-cartpole-policy --iterations 24 --population 32 --elite-count 6 --num-envs 64 --train-steps 256 --eval-steps 1024 --seed 0 --policy-out reports\symliquid_ocean_cartpole_policy_cem_seed0.json --out reports\symliquid_ocean_cartpole_policy_cem_seed0_train.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --artifact reports\symliquid_ocean_cartpole_policy_cem_seed0.json --env ocean-cartpole --rollout-smoke-steps 1024 --num-envs 128 --out reports\puffer_ocean_cartpole_learned_cem_seed0_1024x128.json
```

```text
Algorithm: local cross-entropy search over a SymLiquid-compatible linear CartPole state policy
Feature set: cartpole_linear_v1
Seed 0 train eval mean reward: 0.995
Seed 0 rollout mean reward: 0.995
Seed 0 rollout dones: 44
Seed 0 rollout truncations: 596
Transitions/sec: 36694.1
External inference calls: 0
Policy artifact: reports/symliquid_ocean_cartpole_policy_cem_seed0.json
Train report: reports/symliquid_ocean_cartpole_policy_cem_seed0_train.json
Rollout report: reports/puffer_ocean_cartpole_learned_cem_seed0_1024x128.json
```

Light three-run learned-policy sweep:

```text
Mean reward: 0.99471
Std reward: 0.00070
Mean transitions/sec: 36746.1
External inference calls: 0
Sweep: reports/puffer_ocean_cartpole_learned_cem_seed_sweep_1024x128.json
```

Local Ocean breadth smoke:

```bash
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --train-discrete-policy --env ocean-chain --iterations 4 --population 12 --elite-count 4 --num-envs 64 --train-steps 128 --eval-steps 512 --seed 0 --policy-out reports\symliquid_ocean_chain_policy_cem_prior_seed0.json --out reports\symliquid_ocean_chain_policy_cem_prior_seed0_train.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --train-discrete-policy --env ocean-memory --iterations 4 --population 12 --elite-count 4 --num-envs 64 --train-steps 128 --eval-steps 512 --seed 0 --policy-out reports\symliquid_ocean_memory_policy_cem_prior_seed0.json --out reports\symliquid_ocean_memory_policy_cem_prior_seed0_train.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --train-discrete-policy --env ocean-tmaze --iterations 4 --population 12 --elite-count 4 --num-envs 32 --train-steps 96 --eval-steps 256 --seed 0 --policy-out reports\symliquid_ocean_tmaze_policy_cem_prior_seed0.json --out reports\symliquid_ocean_tmaze_policy_cem_prior_seed0_train.json
cargo build --release -p symliquid-ffi
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --artifact reports\symliquid_ocean_tmaze_policy_cem_prior_seed0.json --env ocean-tmaze --rollout-smoke-steps 512 --num-envs 128 --use-rust-ffi --out reports\puffer_ocean_tmaze_rust_ffi_128x512.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --train-discrete-policy --env ocean-noisy-memory --iterations 8 --population 20 --elite-count 5 --num-envs 64 --train-steps 192 --eval-steps 768 --seed 0 --use-rust-ffi --policy-out reports\symliquid_ocean_noisy_memory_policy_cem_prior_seed0.json --out reports\symliquid_ocean_noisy_memory_policy_cem_prior_seed0_train.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --artifact reports\symliquid_ocean_noisy_memory_policy_cem_prior_seed0.json --env ocean-noisy-memory --rollout-smoke-steps 1024 --num-envs 128 --use-rust-ffi --out reports\puffer_ocean_noisy_memory_rust_ffi_cem_seed0_1024x128.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --train-discrete-policy --env ocean-noisy-memory --iterations 8 --population 20 --elite-count 5 --num-envs 64 --train-steps 192 --eval-steps 768 --seed 0 --use-rust-ffi --policy-out reports\symliquid_ocean_noisy_memory_policy_rust_trainer_seed0.json --out reports\symliquid_ocean_noisy_memory_policy_rust_trainer_seed0_train.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --train-discrete-policy --env ocean-noisy-tmaze --iterations 12 --population 24 --elite-count 6 --num-envs 64 --train-steps 192 --eval-steps 768 --seed 2 --use-rust-ffi --policy-out reports\symliquid_ocean_noisy_tmaze_policy_sum_rust_trainer_seed2.json --out reports\symliquid_ocean_noisy_tmaze_policy_sum_rust_trainer_seed2_train.json
python scripts\benchmark_treadmill.py --reports reports --out reports\benchmark_treadmill_status.json --benchmark-ledger-out reports\benchmark_ledger.json --model-ledger-out reports\model_ledger.json --public-comparator-ledger-out reports\public_comparator_ledger.json
```

```text
Supported local Ocean surfaces: cartpole, chain, memory, tmaze, noisy-memory, noisy-tmaze, slot-tmaze
Policy modes:
  cartpole_linear_v1
  dense_linear_v1
  memory_recurrent_linear_v1
  tmaze_recurrent_linear_v1
  evidence_recurrent_linear_v1
Initialization: CGS governance prior + local CEM tuning
Mean normalized performance across four surfaces: 0.9988
Mean transitions/sec: 51336.7
External inference calls: 0
Breadth report: reports/puffer_ocean_breadth_smoke_cartpole_chain_memory_tmaze.json
```

Rust FFI policy/backend parity:

```text
Crate: crates/symliquid-ffi
Library: target/release/symliquid_ffi.dll
Envs: cartpole, chain, memory, tmaze
Reward parity with Python scorer: true
Speedups: 6.06x, 5.32x, 8.12x, 9.90x
Summary: reports/puffer_ocean_rust_ffi_parity_summary.json
```

Noisy delayed-memory pressure test:

```text
Env: ocean-noisy-memory
Feature set: evidence_recurrent_linear_v1
Policy backend: rust_ffi
Train eval mean reward per transition: 0.07264
Rollout normalized performance: 0.8641
Last-cue baseline normalized performance: 0.7445
Rollout transitions/sec: 204968.7
External inference calls: 0
Policy artifact: reports/symliquid_ocean_noisy_memory_policy_cem_prior_seed0.json
Train report: reports/symliquid_ocean_noisy_memory_policy_cem_prior_seed0_train.json
Rollout report: reports/puffer_ocean_noisy_memory_rust_ffi_cem_seed0_1024x128.json
Baseline report: reports/puffer_ocean_noisy_memory_last_cue_baseline_rust_ffi_1024x128.json
```

Rust FFI breadth smoke:

```text
Envs: cartpole, chain, memory, tmaze, noisy-memory, noisy-tmaze, slot-tmaze
Mean normalized performance: 0.9720
Minimum normalized performance: 0.8641
Mean transitions/sec: 278999.4
External inference calls: 0
Report: reports/puffer_ocean_rust_ffi_breadth_cartpole_chain_memory_tmaze_noisy.json
```

Rust-owned rollout/trainer smoke:

```text
Trainer backend: rust_ffi_rollout_trainer
Envs: chain, memory, tmaze, noisy-memory, noisy-tmaze, slot-tmaze
Mean normalized performance: 0.9658
Minimum normalized performance: 0.8631
Mean rollout transitions/sec: 305117.8
Noisy-memory train wall time observed: 2.31s
External inference calls: 0
Report: reports/puffer_ocean_rust_rollout_trainer_breadth_summary.json
```

Benchmark treadmill:

```text
Policy: local_only_no_external_inference
Methodology: capability_ratchet / benchmaxxing_performance_ratchet
Current active family: coding_local_sandbox
Best public calibration card: source_human_eval, 32 tasks, pass rate 0.78125
Below-floor receiver cards: MBPP 0.6875, EvalPlus 0.59375, BigCodeBench 0.25, LiveCodeBench 0.21875
Next overnight rotation target: source-agnostic type/return-shape, edge-condition, admissibility/interface, and algorithmic-planning pressure selected by transfer generalization audit
Real public code calibration: broad matrix YELLOW, below promotion floor
Broad public pass rate: 0.50625 across 160 public calibration tasks
Required promotion floor: 0.70
Candidate gate: promote=false
Rule: real public code evidence must come from token-level learned student generation, not templates, public-solution leakage, loop-closure tools, task-id lookup, or external inference.
Next action: target transferable type/return-shape, admissibility/interface, edge-condition, and algorithmic-planning residual families with private hidden-test training; BigCodeBench and LiveCodeBench both have 32+ clean calibration tasks now and should be treated as below-floor semantic transfer receivers.
Report: reports/benchmark_treadmill_status.json
Benchmark ledger: reports/benchmark_ledger.json
Model ledger: reports/model_ledger.json
Public comparator ledger: reports/public_comparator_ledger.json
BabyLM residual analysis: reports/babylm_residual_analysis.json
Mutated BabyLM residual analysis: reports/babylm_mutated_residual_analysis.json
Capability ratchet report: reports/capability_ratchet_report.json
Ratcheting Generative Systems audit: reports/ratcheting_generative_system_report.json
Ratcheting Modular Intelligence audit: reports/ratcheting_modular_intelligence_report.json
Octopus Router report: reports/octopus_router_report.json
Arm registry: reports/arm_registry.json
Router eval: reports/octopus_router_eval.json
Routing memory: reports/routing_memory.json
Arm lifecycle ledger: reports/arm_lifecycle_ledger.json
Safety ledger: reports/safety_benchmark_ledger.json
Bridge benchmark ledger: reports/bridge_benchmark_ledger.json
Tool registry: reports/tool_registry.json
Residual escrow: reports/residual_escrow.json
Architecture gate: reports/architecture_gate_report.json
```

BabyLM residual frontier:

```text
Worst field: morphology accuracy=0.8698 residual=0.1302
Worst terms: subject_verb_agreement residual=0.2081, anaphor_agreement residual=0.1111, determiner_noun_agreement residual=0.1108
Worst rules: determiner_noun_agreement_with_adj_irregular_2 residual=0.3250, irregular_plural_subject_verb_agreement_1 residual=0.3235
Recommendation: learned sequence-state features for high-residual BLIMP families, then validate on mutated/private minimal pairs.
Report: reports/babylm_residual_analysis.json
```

Mutated BabyLM holdout regression and frontier:

```text
Factory: scripts/generate_babylm_mutated_holdout.py
Holdout: data/babylm_mutated_holdout_seed49.jsonl
Cases: 4800
Policy: local_only_no_external_inference
Train/eval command: cargo run --release -p symliquid-cli -- train-babylm-probe --input data/babylm_blimp_filtered_train.jsonl --eval-input data/babylm_mutated_holdout_seed49.jsonl --train-limit 53888 --eval-limit 4800 --steps 120000 --hv-dim 8192 --lr 0.08 --stateful --pairwise-contrast --balance-rules --prior-weight 1.0 --out reports/babylm_mutated_holdout_seed49_stateful_grammar_state_frontier.json
Accuracy: 0.9814583
Residual: 0.0185417
Worst terms: ellipsis residual=0.0491, filler_gap_dependency residual=0.0400
Worst rules: wh_vs_that_with_gap residual=0.5000, ellipsis_n_bar_1 residual=0.0501, ellipsis_n_bar_2 residual=0.0481
Residual report: reports/babylm_mutated_residual_analysis.json
Residual escrow: reports/residual_escrow.json
Compiled ratchet workflow: reports/capability_ratchet_run.json

Current code frontier: source-agnostic edge/type/interface pressure
Next code-family rotation target: source-agnostic type/return-shape, edge-condition, admissibility/interface, and BigCodeBench-heavy algorithmic planning pressure after execution-shape private gate cleared and the board receiver run produced only partial broad lift
Current code report: reports/real_code_benchmark_graduation.json
Current broad public calibration pass rate: 0.50625
Current code lifecycle: promotion-facing public calibration, below the 0.70 promotion floor
Next action: keep promotion blocked, continue residual-driven private training, then rerun same-seed calibration
Candidate gate: reports/candidate_promotion_gate.json promote=false
```

Noisy T-maze state-formation result:

```text
Env: ocean-noisy-tmaze
Feature set: evidence_sum_tmaze_recurrent_linear_v1
Architecture change: equal-weight cue accumulation replaced decayed evidence memory.
Prior normalized perf: 0.8070
Current normalized perf: 0.8322
Ceiling-adjusted normalized perf: 0.9943
External inference calls: 0
Report: reports/puffer_ocean_noisy_tmaze_sum_rust_trainer_seed2_rollout_2048x128.json
```

Slot-memory T-maze result:

```text
Env: ocean-slot-tmaze
Feature set: slot_tmaze_recurrent_linear_v1
Capability: two role-filler slots with delayed query at the branch.
Normalized perf: 0.9961
External inference calls: 0
Report: reports/puffer_ocean_slot_tmaze_rust_trainer_seed3_rollout_2048x128.json
```

Noisy-memory ceiling result:

```text
Env: ocean-noisy-memory
Feature set: evidence_sum_recurrent_linear_v1
Ceiling-adjusted normalized eval reward: 0.9975
External inference calls: 0
Report: reports/symliquid_ocean_noisy_memory_policy_rust_trainer_seed4_long_train.json
```

## Promotion Gates

Promote a SymLiquid candidate only if it:

1. improves held-out accuracy or residual over the current local baseline;
2. does not rely on expected outputs, metadata leakage, or external inference;
3. survives at least three seeds;
4. has an ablation showing which component helped;
5. reports failures, not only wins.

## Current Iteration-Speed Cleanup

2026-05-29 checkpoint:

```text
Goal shape: source-level speed/maintainability improvement with compile + audit proof, not automation churn.
Code LM artifact/report helpers: moved under crates/symliquid-cli/src/code_lm_closure/task_features_io/artifact_io.rs.
Training-runtime parent size: task_features_io.rs is 2080 lines, below the 2200-line Rust soft limit.
Verification: cargo check -p symliquid-cli --features cuda passed.
Autonomy cycle split: scripts/autonomy_cycle.py is 1612 lines; runtime helpers live in scripts/autonomy_cycle_runtime.py and source/training inventory command assembly lives in scripts/autonomy_cycle_source_steps.py.
Hive board executor split: scripts/hive_work_board_executor.py is 1578 lines; runtime/reporting/ledger helpers live in scripts/hive_work_board_executor_runtime.py.
Verification: python -m py_compile passed for the split Python modules; autonomy_cycle.py --help and hive_work_board_executor.py --help passed.
Rust benchmark test split: crates/symliquid-core/src/benchmarks.rs is 3461 lines after moving tests into crates/symliquid-core/src/benchmarks/tests.rs.
Verification: cargo check -p symliquid-core and cargo test -p symliquid-core benchmarks::tests -- --nocapture passed.
CUDA rollout feature-support split: crates/symliquid-cuda/src/rollout_cuda.rs dropped from 2866 to 2139 lines after moving retrieval-curriculum row selection, rollout observation encoding, governance priors, label hypervectors, and entity-slot feature helpers into crates/symliquid-cuda/src/rollout_cuda/feature_support.rs.
Verification: cargo check -p symliquid-cuda, cargo check -p symliquid-cuda --features cuda, and cargo check -p symliquid-cli --features cuda passed.
CUDA readout import hygiene: crates/symliquid-cuda/src/readout_cuda.rs now gates CUDA-only imports behind the cuda feature, so the default fast check path is warning-clean while preserving the CUDA hot path.
Benchmark linguistic feature split: crates/symliquid-core/src/benchmarks/linguistic_features.rs dropped from 2609 to 1917 lines after moving pure lexical predicates into crates/symliquid-core/src/benchmarks/linguistic_features/lexicon.rs. The parent now owns feature orchestration and recurrent grammar state; the child owns reusable lexicon/verb/modifier/gap predicates.
Verification: cargo fmt --check, cargo check -p symliquid-core, cargo test -p symliquid-core benchmarks::tests -- --nocapture, git diff --check, attd_analyzer, and system_efficiency_audit passed/refreshed.
Performance optimizer refresh: GREEN, score=0.85, bottleneck_count=0, scheduler_worker_chunks=5, required CUDA eval/readout/rollout chunks planned.
System efficiency audit: YELLOW, finding_count=5, loop_bottleneck_count=2, hard_maintainability_hotspot_count=0, maintainability_hotspot_count=0, maintainability_score=1.0, attd_runtime_overlap_count=0. The active warnings are performance-optimizer worker/resource planning, duplicate live services, root disk floor, and stale historical Code LM wall-clock debt; no new training/public calibration was launched.
ATTD audit: YELLOW, attd_score=0.557193, hard caps passed; linguistic_features.rs dropped out of the top split-large packet after the lexicon split. Remaining pressure is rolling residue and broader training-runtime assembly burden.
Remaining source hotspots under system_efficiency_audit: none.
Policy: no public calibration, model growth, or long training should run from this cleanup evidence alone.
```

## Broad Transfer Residual Decoder Patch

2026-05-29 private-only architecture patch:

```text
Goal shape: residual family -> decoder/ranker/retry behavior -> same-seed private heldout delta.
Source patch: crates/symliquid-cli/src/code_lm_closure/broad_transfer_residual_policy.rs.
Consumers: candidate fanout retry generation, candidate transfer scoring, candidate provenance/report rows.
Residual families routed: edge_case, local_code_generation_adapter_needed, algorithm_choice, type_handling.
Public boundary: retry candidates are private-only and marked non-promotion evidence; public tasks/tests/solutions are not used.
Same-seed A/B: reports/broad_transfer_residual_decoder_ablation.json.
Private heldout result: pass rate 0.875 -> 1.000 (+0.125), no-admissible 0.125 -> 0.000 (-0.125), residual-router task-count delta +6.
Verification: cargo fmt --check, cargo check -p symliquid-cli, cargo test -p symliquid-cli broad_transfer_residual -- --nocapture, release rebuild, and private-only ablation all passed.
Gate state after source patch: decoder_v2_private_ablation_gate and private_public_transfer_proof are YELLOW because canonical closure artifacts are stale relative to the decoder source. Public calibration remains locked until a fresh private closure/fanout with the patched decoder refreshes those gates.
```

2026-05-29 fresh private validation:

```text
Validation slug: private_pressure_private_recovery_broad_transfer_residual_validation_v1.
Critical fix: scripts/code_lm_train_once_fanout.py now enforces a release CUDA feature build, because a timestamp-fresh non-CUDA binary can otherwise reject --use-cuda-readout.
Train-once/fanout result: GREEN, checkpoint_cuda_readout_used=true, backend=rust_cuda_fast_sparse_code_lm_readout, fanout fresh=true, repeated_training_per_candidate_shard=false.
Decoder gate: GREEN with current fingerprint 8936e8594685de2e, private candidates 822, public metadata candidates 55, contract-guided candidates 92, STS-conditioned candidates 267.
Transfer proof: YELLOW, not public-calibration-ready. Actual coverage delta +0.199364 and no-admissible delta -0.191551 passed, but eligible coverage delta +0.000579 missed the +0.03 floor.
Residual packet: reports/private_public_transfer_residual_packet.json names next source patch eligible_receiver_inventory_router_v1 targeting broad_transfer_residual_policy.rs, candidate_fanout/expression_pool.rs, candidate_fanout/task_rows.rs, and contract_verifier/scoring.rs.
Safety: no public tests, public solutions, public calibration, model growth, or candidate promotion ran.
```

Good overnight goals should be framed as bounded engineering outcomes:

```text
Improve one named bottleneck or hotspot.
State the expected file/report change.
Define compile/smoke/audit acceptance criteria.
Define stop conditions that prevent repeated launches.
Update the evidence docs after verification.
```

## Next Steps

1. Move the Rust-owned rollout trainer from CPU FFI to CUDA batched env stepping, reward/done buffers, and optimizer state.
2. Add reservoir-only, VSA-only, and tiny GRU/RNN local baselines on the same public BLIMP split.
3. Replace remaining contract-specific symbolic rules with learned liquid/reservoir/VSA sequence updates.
4. Move from Rust FFI scoring to Rust/CUDA batched env stepping, rewards, dones, and optimizer state.
5. Keep the treadmill active: once BabyLM saturates, add harder local benchmarks before optimizing already-solved surfaces.
6. Add Parameter Golf style BPB/perplexity evaluation if SymLiquid exposes a language-model scoring interface.
