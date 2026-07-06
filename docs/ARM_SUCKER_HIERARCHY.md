# Arm-Sucker Transfer Hierarchy

Project Theseus uses the Octopus router to keep capability modular. An arm is a high-transfer specialist that should help many related tasks. A sucker is a small, loadable specialization attached to an arm for low-transfer details such as one game, simulator, controller schema, local asset contract, or benchmark adapter.

The purpose is to avoid spawning a new top-level arm every time Theseus sees a new game or environment. The head routes to a durable parent arm, the parent arm loads its shared skills, and then only the needed sucker is attached for the current frontier.

## Rule

Use a sucker before creating a new arm when:

- the new work mostly reuses a parent arm's primitives;
- the new work only changes action schema, observation adapter, local assets, controller prior, or benchmark rules;
- permissions and runtime tier match the parent arm;
- the specialization is not yet proven useful across sibling domains.

Create or promote a new arm when:

- the sucker repeatedly transfers to sibling suckers;
- the specialization needs distinct permissions, memory, runtime, or safety gates;
- the parent arm becomes bloated;
- router confusion increases because unrelated tasks share a parent.

## Current Shape

`video_game_play_arm` is the high-transfer game-control core. It owns observation normalization, action mapping, reward/done normalization, replay traces, controller priors, residual escrow, and generic game curriculum logic.

Its suckers include:

- `minecraft_open_world_sucker` for Minecraft-like open-world tasks;
- `crafter_bridge_sucker` for the faster Crafter/Craftax bridge;
- `minecraft_java_local_sucker` for local Java Minecraft integration;
- `emulator_gba_sucker` for BYO-ROM-safe emulator traces.

`drone_racing_control_arm` remains the high-transfer drone-control core. Its suckers include:

- `gym_pybullet_hover_sucker`;
- `pyflyt_waypoint_sucker`;
- `ai_grand_prix_sitl_sucker`.

`language_reasoning_code_arm` owns the current code-family learning pressure.
Its suckers include:

- `python_grammar_sucker` for AST/function-shape validation and sandbox-ready
  code checks;
- `rust_grammar_sucker` for cargo/rustfmt/clippy-style validation;
- `javascript_typescript_grammar_sucker` for parser/type-checker validation;
- `english_surface_sucker` for surface grammar and response-structure checks;
- `sbl_router_sucker` for semantic-backbone routing traces.

These suckers are rule/verifier layers around the learned core. They may reject
invalid structure, route residuals, and condition STS streams, but they must not
contain public benchmark answers or task-id lookup paths.

## Lifecycle / Cell Death

Arms, suckers, verifier suckers, systems, and tools are now governed by
`scripts/cell_lifecycle.py` and `configs/cell_lifecycle_policy.json`. The goal
is anti-bloat and continual pressure:

- useful cells are renewed or protected;
- weak cells become improve candidates;
- bloated cells become split/compress candidates;
- repeated useful workflows create tool/sucker pressure;
- mastered data can be proposed for archival through a visible prune plan.

The lifecycle report is non-destructive by default. It writes
`reports/cell_lifecycle.json` and, when requested,
`reports/cell_lifecycle_prune_plan.json`; it does not delete cells or training
data without an explicit safe policy and human-visible plan.

## Generated Reports

The hierarchy is governed by:

- `configs/arm_sucker_policy.json`
- `scripts/arm_sucker_registry.py`
- `reports/arm_sucker_registry.json`
- `reports/arm_sucker_registry.md`
- `reports/cell_lifecycle.json`

The autonomy loop refreshes the registry before transfer planning. Transfer artifacts include `target_sucker` and `sucker_chain` so pressure runners can load parent skills first and specialization second.

## Transfer Contract

Every graduated sucker should export at least one reusable artifact:

- policy prior;
- replay or trace capsule;
- normalized state/action schema;
- residual curriculum;
- verification contract.

Future siblings can load those artifacts through the parent arm. If a sucker repeatedly helps sibling suckers, it becomes a promotion candidate for a full arm.
