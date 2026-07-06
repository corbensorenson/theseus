# Cognitive Loop Closure

## Compiling Repeated AI Reasoning Trajectories into Verified Parameterized Tools

**White Paper**  
**Public Release v1.0 - May 2026**  
**Author:** Corben Sorenson  
**Status:** Conceptual Framework + Agent Architecture Proposal

## TL;DR

Modern AI agents repeatedly perform similar chains of reasoning, tool use,
retrieval, editing, verification, and correction. Most still treat each task as
a blank-slate reasoning problem. This wastes inference compute, increases
behavioral variability, hides failure modes, and prevents durable procedural
skill accumulation.

**Cognitive Loop Closure** is a metacognitive architecture in which an agent
reviews safe execution history, detects recurring reasoning/action trajectories,
abstracts invariant structure, discovers parameters, synthesizes deterministic
or bounded tools, verifies those tools, and routes future matching tasks through
procedural memory.

Core thesis:

```text
Repeated cognition should become procedural memory.
```

Formal sketch:

```text
{tau_1, tau_2, ..., tau_n} -> T_phi(p)
```

Many repeated trajectories become one callable tool with parameters. This is
not caching, prompt templating, or fine-tuning. A cache stores answers for
repeated inputs. A prompt template still asks the model to improvise.
Fine-tuning hides skill in weights. Loop Closure externalizes skill into typed,
testable, versioned, monitored, governed tools.

## Three Execution Modes

| Mode | Purpose |
| --- | --- |
| Interpreter mode | Slow, flexible reasoning for novel or ambiguous tasks. |
| Compiled-tool mode | Verified tool execution for repeated, well-understood tasks. |
| Reflex/failsafe mode | Immediate safety behavior when reasoning is too slow or risk boundaries are approached. |

The third mode matters. A drone, robot arm, production deployer, or transaction
system cannot always fall back to a slow reasoning loop. Safety-critical systems
need reflexive controllers, runtime monitors, control barriers, hard stops,
holds, emergency landing, or isolation policies.

## Ten Components

| Component | Role |
| --- | --- |
| Trajectory logger | Records safe execution traces without private chain-of-thought. |
| Loop detector | Finds recurring task/action/tool/verification patterns. |
| Abstraction engine | Separates invariant procedure from variable parameters. |
| Active parameter discovery | Probes candidate tools to reveal hidden assumptions. |
| Tool synthesizer | Creates deterministic or bounded procedures. |
| Verifier | Tests prior, held-out, synthetic, adversarial, and sandbox cases. |
| Tool registry | Stores tool cards, schemas, risk tiers, provenance, and metrics. |
| Router | Chooses interpreter, compiled-tool, reflex/failsafe, clarification, or refusal. |
| Runtime monitor | Tracks live behavior, residuals, drift, and safety boundaries. |
| Revision/retirement manager | Updates, merges, disables, or retires tools as environments change. |

## Current Rust Surface

The implementation lives in:

```text
crates/symliquid-core/src/loop_closure.rs
crates/symliquid-core/tests/test_loop_closure.rs
```

It currently includes:

- `TrajectoryRecord`: safe execution trace metadata with parameters,
  environment metadata, residuals, risk, latency, and safety-event flags.
- `detect_loop_candidates`: groups repeated verified trajectories.
- `LoopCandidate`: recurrence, invariant steps, parameters, residuals, savings.
- `infer_parameter_states`: classifies observed variables as parameter,
  precondition, or unknown assumption.
- `active_parameter_probe_plan`: emits counterfactual, synthetic, adversarial,
  environment, and sandbox probes for a candidate loop.
- `synthesize_tool_card`: creates a versioned procedural-memory card.
- `ToolRegistry::route`: legacy simple routing by family, parameters, and risk.
- `ToolRegistry::route_execution`: richer interpreter/tool/reflex routing.
- `RiskTier`, `VerificationGrade`, `RuntimeTier`, `LatencyClass`,
  `ExecutionMode`, and `ParameterState`.

## Formal Model

A trajectory records what the agent safely did:

```text
tau_i = (x_i, c_i, a_0:n, o_0:n, y_i, v_i)
```

| Symbol | Meaning |
| --- | --- |
| `x_i` | Initial task or request. |
| `c_i` | Context. |
| `a_0:n` | Actions, tool calls, edits, searches, summarized decisions. |
| `o_0:n` | Observations, tool results, artifacts, feedback. |
| `y_i` | Final output. |
| `v_i` | Verification outcome. |

A closed loop tool is:

```text
T_phi(p) -> y
```

Future tasks are routed by:

```text
if match + preconditions + risk + verification pass:
    execute compiled tool
elif safety-critical latency dominates:
    trigger reflex/failsafe
else:
    reason in interpreter mode
```

## Active Parameter Discovery

Passive abstraction is not enough. If every observed repository used `cargo`,
the agent should not conclude that `cargo test` is globally invariant. The true
abstraction may be "run the repository's test workflow", with package manager,
test scope, environment variables, timeout, and known flaky tests as parameters.

Parameter discovery methods:

- Historical variance analysis.
- Counterfactual replay.
- Synthetic case generation.
- Adversarial edge-case probing.
- Environment interrogation.
- Human or supervisory questioning.

Variables should be classified as:

| State | Meaning |
| --- | --- |
| Invariant | Should not change across valid uses. |
| Parameter | Expected to vary and should be exposed in the schema. |
| Precondition | Must hold before execution. |
| Unknown assumption | Suspected dependency requiring more evidence. |

## High-Bandwidth Embodied Logging

Embodied agents cannot store every frame, lidar packet, IMU sample, motor
command, and controller state forever. Loop Closure for physical agents needs
hierarchical logging:

| Log type | Purpose |
| --- | --- |
| Raw telemetry | High-fidelity replay around failures and representative cases. |
| Cognitive trajectory | Compressed semantic/action trace for loop detection. |
| Event log | Obstacle, slip, tool activation, failsafe trigger, verifier event. |
| Residual log | Prediction failures, anomalies, monitor violations. |

Use triggered retention around important events:

```text
[t_event - delta_pre, t_event + delta_post]
```

Retain richer raw windows for near collisions, high uncertainty, controller
saturation, human override, verifier failure, reflex/failsafe activation, and
large residuals.

Memory budget:

```text
B_raw + B_events + B_features + B_residuals <= B_budget
```

The design goal is to log enough structure to discover loops, enough detail to
debug failures, and enough safety evidence to audit reflex behavior.

## Tool Cards

Every closed loop should be stored as a governable artifact.

| Field | Purpose |
| --- | --- |
| Tool name | Human-readable identifier. |
| Purpose | What the tool does. |
| Task family | What class of tasks it covers. |
| Inputs / outputs | Typed objects. |
| Parameters | Variables that customize execution. |
| Hidden assumptions | Dependencies not yet proven safe. |
| Active probes | Perturbations used to test the abstraction. |
| Preconditions / postconditions | Required before and after execution. |
| Verification grade | How well the tool has been tested. |
| Runtime tier | Where the tool executes. |
| Latency class | How quickly it must respond. |
| Risk tier | Consequence of failure. |
| Allowed side effects | What the tool may mutate. |
| Permissions | Capabilities granted. |
| Runtime monitors | Live checks during execution. |
| Fallback mode | Interpreter, alternate tool, human review, or reflex/failsafe. |
| Provenance | Source trajectories. |
| Metrics | Success rate, savings, failures, residuals. |
| Retirement criteria | When to disable or replace it. |

## Verification Grades

| Grade | Meaning |
| --- | --- |
| Unverified | Candidate tool; cannot route automatically. |
| Replay-passed | Reproduces prior successful trajectories. |
| Held-out-passed | Handles unseen prior examples. |
| Synthetic-passed | Handles generated variants. |
| Adversarial-tested | Known failure boundaries explored. |
| Runtime-monitored | Live execution is monitored. |
| Human-approved | Approved for a specified risk tier. |
| Certified | Meets domain-specific assurance requirements. |

Verification is not absolute in open worlds. The defensible claim is that Loop
Closure makes repeated procedural skill more verifiable than unconstrained
repeated reasoning.

## Runtime Tiers

| Tier | Environment | Appropriate for |
| --- | --- | --- |
| E0 | Text template | Low-risk drafting and formatting. |
| E1 | Structured workflow | Human-reviewed procedures. |
| E2 | Typed function | Deterministic digital transformations. |
| E3 | Sandboxed runtime | Untrusted or generated code. |
| E4 | Memory-safe systems runtime | Higher-assurance tools. |
| E5 | Real-time embedded/reflex runtime | Safety-critical embodied agents. |

Tools should run in the least powerful environment sufficient for the task,
with typed schemas, capability restrictions, time limits, memory limits, and
dry-run modes where appropriate.

## Lifecycle

```text
Candidate -> Proposed -> Probed -> Tested -> Probationary -> Active -> Monitored -> Revised -> Retired
```

Close a loop only when expected value exceeds total lifecycle cost:

```text
Close(T) = [F * DeltaC * Q * A > C_T + M_T + R_T + V_T + D_T]
```

| Term | Meaning |
| --- | --- |
| `F` | Expected recurrence frequency. |
| `DeltaC` | Expected cost reduction per use. |
| `Q` | Expected quality/reliability improvement. |
| `A` | Automation appropriateness. |
| `C_T` | Tool creation cost. |
| `M_T` | Maintenance cost. |
| `R_T` | Risk cost. |
| `V_T` | Verification cost. |
| `D_T` | Drift and depreciation cost. |

## CGS Mapping

Within Compact Generative Systems, a closed loop tool is a compact generative
structure distilled from repeated behavior.

| CGS component | Loop Closure equivalent |
| --- | --- |
| Seed | Recurring task pattern. |
| Rule system | Deterministic or bounded procedure. |
| Memory/state | Prior trajectories, registry, learned parameters. |
| Residual | Failures, edge cases, corrections, drift. |
| Verification | Tests, schemas, runtime monitors, approvals. |
| Governance | Future routing through tool instead of fresh reasoning. |

Loop Closure is the CGS mechanism by which repeated agent behavior becomes
reusable procedural structure.

## Claims

- AI agents often repeat reasoning/action trajectories.
- Repeated trajectories can be compressed into parameterized tools.
- Tool creation should be driven by recurrence, value, risk, and verification.
- The abstraction engine must actively probe for hidden parameters.
- The router must know when not to use a tool.
- High-stakes or real-time systems need reflex/failsafe modes.
- Tool execution environments should enforce schemas, permissions, boundaries,
  and monitoring.
- High-bandwidth embodied agents require hierarchical logging.

## Non-Claims

- Not every repeated action should be automated.
- Deterministic tools should not replace reasoning.
- Closed tools are not automatically safe.
- Verification is not absolute in open worlds.
- Prompt templates are not sufficient.
- Fine-tuning is not obsolete.
- Current agents do not already implement the full lifecycle.
- Physical systems cannot safely rely on slow reasoning under all conditions.

## Compact Manifesto

An AI should not think through the same staircase forever.

If a path is walked often enough, the path should become a tool.

If the tool works, verify it.

If it fails, expose the residual.

If the residual repeats, revise the tool.

If the world changes, retire it.

If the situation is novel, reason.

If the loop is closed, execute.

If safety is at stake and time is short, reflex.

The goal is not to remove reasoning.

The goal is to reserve reasoning for what is still genuinely novel.

## Selected References

- Sutton, Precup, and Singh, *A Framework for Temporal Abstraction in Reinforcement Learning*.
- van der Aalst, Weijters, and Maruster, *Workflow Mining: Discovering Process Models from Event Logs*.
- Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools*.
- Qian et al., *CREATOR: Tool Creation for Disentangling Abstract and Concrete Reasoning of Large Language Models*.
- Wölflein et al., *LLM Agents Making Agent Tools*.
- Wang et al., *Voyager: An Open-Ended Embodied Agent with Large Language Models*.
- Agostinelli et al., *SmartRPA: Generating Software Robots from User Interface Logs*.
- Billard et al., *Robot Programming by Demonstration*.
- Haith and Krakauer, *The Multiple Effects of Practice: Skill, Habit, and Reduced Cognitive Load*.
- Rust Project, *The Rust Programming Language: Ownership*.
- WebAssembly Project, *WebAssembly Overview and Security Model*.
- Sánchez et al., *A Survey of Challenges for Runtime Verification from Advanced Application Domains*.
- Ames et al., *Control Barrier Functions: Theory and Applications*.
