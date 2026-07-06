# Benchmaxx Curriculum

The Benchmaxx curriculum is the long-range course map for growing SparkStream/RMI into a broader agentic local system. It makes the next capability lane explicit so the teacher does not have to invent the path on every wall.

Machine-readable source of truth:

- `configs/benchmaxx_curriculum.json`
- `reports/benchmaxx_curriculum.json`
- `reports/benchmaxx_curriculum.md`

Refresh it with:

```powershell
py -3.13 scripts\benchmaxx_curriculum.py --out reports\benchmaxx_curriculum.json --markdown-out reports\benchmaxx_curriculum.md
```

The autonomy loop refreshes it every cycle and exposes it through the dashboard API.

## Course

The current planned course is:

1. SymLiquid substrate and CGS governance.
2. BabyLM grammar and mutated linguistic generalization.
3. Residual-targeted synthetic data and loop closure.
4. Local RL memory, control, exploration, and credit assignment.
5. Minecraft/open-world agent pressure.
6. Drone racing and simulator control.
7. Emulator RL from user-owned ROMs.
8. Language, reasoning, and coding benchmarks.
9. Tool-use and multi-turn dialogue agents.
10. Browser and web agent benchmarks.
11. Desktop/OS computer-use agents.
12. Native voice I/O and multimodal interaction.
13. End-to-end autonomous user agent behavior.

Current temporary focus, 2026-05-18: the curriculum was deliberately set to
`conversation_multiturn_before_returning_to_code_temporarily`. The first smoke
surface saturated, so the conversation gate now requires a large calibration
before graduation. The latest large conversation run passed and is treated as a
regression surface, allowing the scheduler to rotate back to transferable code
semantics while preserving conversation/personality coverage.

The coding stage is intentionally broad now, so the teacher does not have to
discover coding pressure during an autonomy wall. Smoke-passed or staged cards
cover function-level code generation, repository issue repair, coding-agent
harnesses, terminal tasks, polyglot repository work, and governed
competitive-programming data metadata. Full container-backed harnesses use
Docker or Podman when present; otherwise they remain source-contract pressure
with explicit residuals.

The coding lane now has two movement rules:

- same-family rotation first among runnable code cards;
- transfer interleave if the public-transfer floor remains blocked long enough.

Transfer interleave is not promotion evidence. It is a way to train somewhere
else, export residual/transfer artifacts, and return to code with broader
structure loaded. Current configured interleaves include local RL
memory/control, broad transfer evals, multi-turn conversation/personality
continuity, web-agent fixtures, and smoke-passed drone control. The teacher
remains architecture guidance only and must not provide benchmark answers.

Each stage records:

- capability goal;
- benchmark name patterns;
- source catalog IDs;
- promotion gate;
- next frontier family;
- teacher policy;
- asset or license blockers.

## Rules

The curriculum follows the RMI rules:

- Mastered benchmarks become regression surfaces.
- Unsolved tails enter residual escrow.
- Active frontiers are not rerun forever after saturation.
- New frontiers are pulled from the next curriculum lane.
- The teacher is audit/proposal only unless local evidence shows a real architecture wall.
- External inference remains forbidden outside `scripts/teacher_oracle.py`.
- Bulk datasets, uncertain licenses, and commercial ROM downloads require explicit approval.

## Native Voice Rule

Voice is part of the Theseus head/router I/O boundary. Benchmarks and licensed
audio data may apply pressure, and ordinary audio codecs/loaders may be used as
non-inference plumbing, but STT/TTS capability must be learned by Theseus-native
components. Installing or calling Whisper, Vosk, SpeechBrain pretrained models,
pyttsx3, cloud speech APIs, or any other borrowed speech inference does not
count as progress and must not satisfy voice gates.

`configs/native_voice_training_policy.json` and
`scripts/native_voice_training_manifest.py` now make that lane automatic:
LibriSpeech, LibriTTS, LJSpeech, Common Voice, and VCTK are assigned explicit
STT/TTS roles, license gates, storage caps, and tiny-shard rules. The autonomy
loop refreshes `reports/native_voice_training_manifest.json` before scoring
`reports/native_voice_io.json`, so voice can become a future frontier without
asking the teacher to rediscover speech data sources every cycle.

## Conversation And Personality Rule

Theseus must be able to hold a conversation as well as do work. The
Tool-Use and Multi-Turn Dialogue Agents stage now includes the local
`conversation_multiturn` surface from
`reports/multi_turn_conversation_benchmark.json`. That benchmark runs the live
`checkpoint_chat` path through multi-line, multi-turn cases for memory,
correction handling, constraint carry, handoff continuity, and personality-core
attachment.

The benchmark is not a claim that the final learned conversational model is
done. It is a runtime pressure surface and regression guard. It keeps the
personality core integrated while the broader learner grows, and it gives the
hive/mobile chat path a concrete continuity contract.

Latest checked conversation lane, 2026-05-18:

```text
high_transfer_multi_turn_conversation: GREEN
suite mode: large
cases: 72
turns: 152
accuracy: 0.968
passed cases: 72/72
personality-ready turns: 152/152
graduated: true
saturated: true
personality_runtime_audit: GREEN
drift_average_score: 1.000
open conversation private SFT rows: 972
open conversation STS rows: 972
```

The open conversation pantry is private pressure only. It is not public
promotion evidence, uses governed tiny/capped samples, rejects public code-eval
overlap tokens, and does not use teacher distillation. Once the large 64+ case
conversation gate clears the configured accuracy floor, both the conversation
benchmark and conversation pantry become regression-only in
`reports/high_transfer_curriculum_scheduler.json`; they should not keep
consuming critical frontier slots unless a regression or new residual appears.
The scheduler chooses the strongest conversation evidence by graduation and
case count, so a newer small smoke report cannot erase an earlier large-suite
graduation.

## Dashboard Semantics

`reports/benchmaxx_curriculum.json` exposes:

- `summary.current_stage_id`: the first stage that is not locked as regression.
- `summary.next_frontier_family`: the frontier family the autonomy loop should rotate toward.
- `next_frontier.runnable_now`: whether current local assets/adapters are enough to run it now.
- `next_frontier.same_family_rotation`: why a code card was kept or rotated.
- `next_frontier.transfer_interleave`: whether broader transfer pressure should
  temporarily replace the code runner, what to return to, and when teacher
  architecture guidance becomes appropriate.
- `near_term_queue`: the active or blocked stages most relevant to the next few cycles.

The dashboard surfaces the same report under `/api/status` as `benchmaxx_curriculum`.

## Teacher Role

The teacher should not be the system's permanent planner. The teacher is used sparingly to:

- audit whether the system is staying on course;
- diagnose repeated residual clusters;
- propose architecture changes after the intervention ladder reaches that level;
- review adapter or benchmark-design walls.

Ordinary benchmark selection, frontier rotation, residual escrow, regression promotion, source catalog staging, and local training remain autonomous local machinery.
