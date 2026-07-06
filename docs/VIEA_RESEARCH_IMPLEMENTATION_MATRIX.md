# VIEA Research Implementation Matrix

This matrix tracks how the recent planning, memory, tool-use, and deterministic
solver research is being ported into Theseus. It is intentionally terse; report
JSON files are the evidence source.

| Research Area | Theseus Surface | Implemented Now | Remaining Gap |
| --- | --- | --- | --- |
| VIEA | `scripts/viea_execution_spine.py`, `scripts/viea_artifact_kernel.py` | Command contracts, artifact objects, claim refs, evidence refs, support states, residuals, training evidence, procedural tools. | Promotion remains limited to deterministic private execution until broader verifiers pass. |
| PlanForge / LLMCompiler | `scripts/theseus_plan_compiler.py`, `configs/theseus_plan_compiler.json` | DAGs, critical-path schedule, semantic hashes, VCM context slices, execution packets, private execute-mode bridge. | Parallel learned-tool execution and full WorkBoard replacement are not claimed. |
| LangGraph-style durability | `scripts/viea_execution_spine.py`, `reports/viea_execution_spine.sqlite` | Run ids, node leases, retry policy, stale lease recovery, checkpoints, resume/cancel hooks. | UI-level human interrupts are not wired yet. |
| MCP | `reports/deterministic_tool_registry.json` | Tool cards include schema, trust tier, auth scope, side effects, VCM bindings, replay checksum, failure states. | External MCP connector bridge is still pending; arbitrary remote shell remains blocked. |
| Memory control | `reports/viea_execution_spine_vcm_artifacts.jsonl`, `scripts/virtual_context_memory.py` | Executed tool nodes emit VCM/evidence artifacts with context hashes and support states. | Native KV/prefix-cache parity is not claimed. |
| Deterministic solvers | `scripts/theseus_deterministic_tool_substrate.py`, `configs/deterministic_tool_substrate.json` | SymPy, SciPy, mpmath, interval arithmetic, linear algebra, Lean, Z3, bounded equality saturation, BM25, hybrid local search, VCM search. | Equality saturation is bounded/local, not a production e-graph optimizer claim. |
| Toolformer / BFCL | `reports/viea_tool_use_learning_traces.jsonl`, `reports/viea_tool_use_training_evidence.jsonl` | Private tool-use traces and governed-admission evidence capture selected tool, args hash, result state, verifier outcome, outcome label, support state, and eligibility. | Public BFCL remains calibration-only; smoke does not silently write durable training rows. |
| Loop closure | `reports/viea_execution_loop_closure_candidates.json`, `reports/viea_verified_procedural_tools.json` | Repeated verified traces become procedural tool records with preconditions, postconditions, tests, risk, provenance, update, and retirement policy. | One-off successful tools still require repetition before promotion. |
| Planning and agent benchmarks | `configs/viea_execution_spine.json`, `reports/viea_execution_spine.json` | Private dry-run scorers for PlanBench, TravelPlanner, BFCL/Gorilla, WebArena/BrowserGym, OSWorld, and TheAgentCompany. | No public calibration is run here. |

No-cheat boundary: public benchmarks are calibration-only; external inference
calls are zero; fallback returns are not allowed; unsupported tools emit
structured non-solved states.
