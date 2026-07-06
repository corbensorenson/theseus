# Reality Manipulator MVP

The Reality Manipulator is now represented in the local Theseus runtime as a
deterministic intent-to-artifact compiler. It is not VR, fabrication, chip
deployment, or proof of ASI. It is the first practical kernel for the idea that
human intent should become a durable, gated project world instead of vanishing
inside a chat transcript.

Run it with:

```powershell
python scripts\reality_manipulator.py --out reports\reality_manipulator.json --markdown-out reports\reality_manipulator.md --bundle-dir reports\reality_manipulator\latest_world
```

The default acceptance scenario is:

```text
Create a world for a modular fabrication-aware AI assistant that can design
small physical tools and compile embedded software for them.
```

## Loop

The implemented loop is:

```text
raw goal
  -> first-class VIEA command contract
  -> eight-limb Grimoire spell
  -> private Portal world
  -> artifact graph
  -> claim ledger
  -> critique log
  -> specialist router arms
  -> compile targets
  -> release gates
  -> residual escrow
  -> primitive candidates
  -> specialist lifecycle
  -> workflow-to-tool metrics
  -> feedback plan
```

This is the MVP vertical slice from the broader Reality Manipulator concept.
Every meaningful object in the report is meant to map to a future Portal object,
dashboard object, Genesis artifact, or release-gate input.

The canonical architecture framing is [Verified Intent-To-Execution
Architecture](VIEA.md). Reality Manipulator is its current vertical MVP.

## Spell Contract

The canonical spell remains the eight-limb contract:

| Limb | Purpose |
| --- | --- |
| Role | What expertise is invoked. |
| Objective | What outcome matters. |
| Context | What is true in the current world. |
| Constraints | What may not break. |
| Procedure | How the work proceeds. |
| Output Contract | What artifact shape must be returned. |
| Verification | How correctness is checked. |
| Failure Behavior | How uncertainty degrades safely. |

The compiler also runs coil inspection across Objective x Verification,
Context x Constraints, Procedure x Output, Constraints x Failure Behavior, and
Role x Output.

## Runtime Boundaries

Compile targets are modeled explicitly:

- portal: private spatial world and artifact objects;
- digital: specs, codebases, dashboards, datasets, workflows;
- chip: firmware, embedded builds, GPU kernels, FPGA assumptions;
- matter: CAD packages, DFM risks, BOMs, fabrication packets, inspection plans;
- robotic: motion plans, controllers, failsafes, telemetry loops;
- organizational: workflows, policies, roles, release rituals.

High-risk targets such as chip, matter, and robotic runtimes are planning-only
until their required gates pass. The core rule is:

```text
nothing leaves a world into shared reality without the right gate
```

## Reports

The compiler writes:

| Artifact | Purpose |
| --- | --- |
| `reports/reality_manipulator.json` | Canonical machine-readable report. |
| `reports/reality_manipulator.md` | Human-readable summary. |
| `reports/reality_manipulator/latest_world/world.json` | World metadata, imports, arms, targets, benchmarks, residuals, permissions. |
| `reports/reality_manipulator/latest_world/command_contract.json` | First-class structured command contract. |
| `reports/reality_manipulator/latest_world/artifacts.json` | Artifact graph nodes and edges. |
| `reports/reality_manipulator/latest_world/claim_ledger.json` | Claims with support state and risk. |
| `reports/reality_manipulator/latest_world/critique_log.json` | Open critiques and required revisions. |
| `reports/reality_manipulator/latest_world/structured_output.md` | Human-readable implementation handoff/spec. |
| `reports/reality_manipulator/latest_world/release_manifest.json` | Internal world snapshot manifest. |
| `reports/reality_manipulator/latest_world/primitive_registry.json` | Candidate reusable primitives extracted from the world. |
| `reports/reality_manipulator/latest_world/specialist_lifecycle.json` | Lifecycle governance rows for selected arms. |
| `reports/reality_manipulator/latest_world/workflow_tool_metrics.json` | Loop-closure/tool-compiler metrics and guardrails. |
| `reports/reality_manipulator/latest_world/feedback_plan.md` | What reality should teach the next iteration. |
| `reports/viea_report_map.json` | Machine-readable map from reports/dashboard surfaces to VIEA subsystems. |
| `reports/viea_artifact_kernel.sqlite` | SQLite VIEA object store built from the world bundle and related reports. |
| `reports/viea_artifact_kernel.json` | Human/machine view over the SQLite object store. |
| `reports/viea_command_executor.json` | Runnable command-contract route plan, specialist calls, gates, digital packets, and residuals. |
| `reports/digital_runtime_adapter.json` | Digital-first runtime readiness over code patch, test, release, rollback, repo trace, and dashboard packets. |
| `reports/feedback_ratchet.json` | What improved, regressed, became a tool/residual, should expire, and should train next. |

## Integration

`scripts/run_training_ratchet_profile.py` now refreshes the Reality Manipulator
after the Genesis snapshot. `scripts/autonomy_watchdog.py` checks that the world
kernel is fresh, has no failed hard gates, and has zero high-risk approvals
without a gate. `scripts/learning_scoreboard.py` surfaces the report as
artifact-substrate evidence only; it is not student learning evidence.

Genesis also imports `reports/reality_manipulator.json` as an optional source
report so future release bundles can link the world kernel into the broader
artifact graph.

Student learning remains separate. Reality Manipulator/VIEA evidence proves
artifact preservation, routing, gates, and feedback structure; broad public
transfer reports prove whether the learned student is improving.
