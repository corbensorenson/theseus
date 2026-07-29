# SparkStream Autonomy

SparkStream is the local bounded-automation and observability shell for
Project Theseus. It reads canonical state, launches registered work when
explicitly authorized, records effects, and exposes local operator controls.

It does not own model capability, training authority, teacher/network
permission, or route truth. Current facts are in
[Project State](PROJECT_STATE.md).

## Current Posture

As of 2026-07-29:

- local authority surfaces pass the bounded security test package;
- teacher and network authority are request-local and default false;
- dashboard and OpenAI-compatible endpoints remain loopback-only;
- long neural training is held;
- KERC/ANE and broad autonomous exploration are not immediate work;
- genuine low-risk dogfood is allowed;
- LAN/public exposure, arbitrary remote execution, and unbounded self-update
  are forbidden.

## Responsibilities

SparkStream may:

- read current reports and registered state;
- run bounded local maintenance and diagnostic commands;
- enforce resource and stop policies;
- maintain explicit job state;
- queue governed teacher requests without executing them;
- record attempted and observed effects;
- write checkpoint and action receipts;
- surface honest blockers;
- support local dogfood and operator inspection.

SparkStream may not:

- infer teacher or network authority from a previous request;
- bypass the runtime training hold;
- consume a frozen capability surface outside its registry;
- execute arbitrary shell or remote work;
- train on public benchmark payloads;
- assign learned-model credit to tools, retrieval, templates, or rules;
- create a new lane because a current owner is blocked;
- expose itself beyond loopback based only on local tests.

## Control Flow

```text
operator request
  -> strict request validation
  -> request-local authority
  -> registry and policy lookup
  -> bounded plan
  -> resource and concurrency checks
  -> registered execution or explicit refusal
  -> attempted-effect record
  -> observed-effect record
  -> verification
  -> rollback/failure boundary
  -> dogfood or maintenance outcome
```

No step inherits authority silently.

## Dashboard

The dashboard is a local operator view. It can show:

- project and route health;
- bounded job state;
- training hold and checkpoint custody;
- resource observations;
- teacher/data accounting;
- VCM and VIEA evidence;
- Hive local status;
- dogfood outcomes;
- generated reports and blockers.

Start only through the supported platform launcher or the registered local
entrypoint. Keep the bind address loopback and use the generated authentication
token.

Security requirements:

- authenticated reads and mutations;
- exact Origin and CSRF checks;
- strict JSON and bounded bodies;
- rate, concurrency, job, and SSE limits;
- sanitized errors;
- random identifiers;
- no wildcard CORS with a tokenless server.

The dashboard does not turn a GREEN report into permission.

## One Bounded Cycle

Use a smoke or maintenance profile only when its registered policy permits it.
Before execution, inspect:

- current project state;
- requested profile;
- teacher/network flags;
- training hold;
- frozen evaluation consumption state;
- active accelerator jobs;
- disk transaction headroom;
- route evidence.

An omitted authority field means false.

## Teacher Requests

The teacher is OpenAI-only and training-time only.

SparkStream may prepare or queue a compact request containing:

- the exact unresolved wall;
- relevant local evidence;
- forbidden actions;
- a bounded requested diagnosis;
- expected machine-readable experiment or patch proposal.

Execution requires explicit request-local authority and the governed teacher
gate. Teacher output is never served to the user and never applied directly.

Bulk generation, public benchmark solving, provider substitution, and
unretained rows are forbidden.

## Network Use

Network access is separate from teacher access. A request must name the
specific bounded network purpose and pass source/license policy.

Allowed examples may include:

- retrieving an explicitly approved public source manifest;
- checking a registered release;
- fetching a bounded licensed sample through the corpus gate.

Broad crawling, bulk downloads, remote execution, credential discovery, and
implicit network inheritance are forbidden.

## Training

SparkStream is not the authority for the active long campaign. Follow
[Real Training Preflight](REAL_TRAINING_PREFLIGHT.md).

While `TRAINING_HELD`:

- no long segment;
- no candidate or seed sweep;
- no KERC/ANE continuation;
- no architecture mutation;
- no frozen functional evaluation.

Low-cost maintenance, documentation, CI, exact hash checks, and genuine
assisted dogfood remain allowed.

## Dogfood

Dogfood is useful only when it represents real work.

Record:

- task class;
- assisted lane;
- redacted intent summary;
- artifacts used;
- one outcome from accepted, missed, ignored, corrected, completed, failed,
  or abstained;
- error family;
- duration bucket;
- external inference count;
- public training row count;
- learned-model credit `false` for assisted behavior.

Do not store raw user text as a training row.

## Hive

SparkStream may submit only registered bounded Hive task kinds. The current
qualified posture is loopback/private and sandbox-fail-closed. See
[Theseus Hive](THESEUS_HIVE.md).

No arbitrary shell, teacher call, git push, ROM import, bulk download, or
untrusted-code execution is a valid Hive task.

## Resource Policy

Resource decisions use causal measurements:

- process and accelerator observations;
- swap growth;
- predicted exhaustion;
- thermal state when available;
- two complete measured checkpoint transactions for disk;
- atomic write and checkpoint behavior.

Do not add clock-of-day windows, arbitrary free-memory floors, or generic
“overnight” semantics.

## Failure Behavior

On failure:

1. stop the bounded owner;
2. preserve partial and pre-failure state;
3. record the exact fault and authority used;
4. distinguish refusal, failure, and incomplete work;
5. avoid automatic widening or retry loops;
6. repair the canonical owner;
7. keep capability and support states unchanged.

## Related Documents

- [Project State](PROJECT_STATE.md)
- [Roadmap](../roadmap.md)
- [Glossary](GLOSSARY.md)
- [VIEA](VIEA.md)
- [Real Training Preflight](REAL_TRAINING_PREFLIGHT.md)
- [Theseus Hive](THESEUS_HIVE.md)
- [Data And Artifacts](DATA_AND_ARTIFACTS.md)

Historical unattended-operation commands are preserved under
`docs/archive/autonomous_weeks_runbook.md`. They are not current authority.
