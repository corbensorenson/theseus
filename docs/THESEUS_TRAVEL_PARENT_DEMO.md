# Project Theseus Travel Demo

Audience: curious non-specialists.

Purpose: show the project honestly without long training, capability-surface
consumption, external inference, LAN exposure, or resource stress.

## Demo Promise

Show four things:

1. Theseus has a clear private/local mission.
2. It preserves exact model and experiment custody.
3. It separates infrastructure health from learned capability.
4. It records failures and blocks unsupported claims.

Do not present Theseus as AGI, ASI, a production assistant, or a model that has
already beaten its controls.

## Current Honest State

Say:

> Theseus is a local AI research system with an exact resumable 57M-parameter
> training campaign, strong evidence and safety machinery, and a real assisted
> runtime. Training is currently held, the model has not consumed its frozen
> functional evaluation, the matched controls are untrained, and daily
> usefulness is not yet proven.

## Five-Minute Demo

From the repository root:

```bash
git status --short
python3 scripts/theseus_project_registry.py --gate
python3 scripts/roadmap_implementation_gate.py --gate
```

Open:

- `docs/PROJECT_STATE.md`;
- `roadmap.md`;
- `docs/GLOSSARY.md`.

Explain:

- registry GREEN means the project’s canonical routes and evidence are
  internally consistent;
- roadmap YELLOW means some phases remain partial or frozen;
- neither is a model-capability score;
- the active model is `TRAINING_HELD` and `NOT_EVALUATED`.

Then show the public reproducibility capsule:

```bash
python3 scripts/public_reproducibility_capsule.py \
  --out-dir /tmp/theseus-public-repro-demo \
  --gate
```

Explain that it proves exact evidence protocol on a tiny authored fixture, not
useful model capability or production speed.

## Assisted Product Demo

Use one low-risk local CLI request that can be checked immediately, such as:

- summarize the current training hold from named local artifacts;
- locate the canonical owner of a project term;
- explain a GREEN/YELLOW gate without changing state;
- identify one safe next action from the roadmap.

Record the outcome honestly. If it misses the request, show that the miss is
useful product evidence rather than hiding it.

Do not award learned-model credit when VCM, tools, reports, rules, retrieval, or
templates contributed.

## What Not To Run

- long or resumed neural training;
- KERC, ANE, optimizer, or architecture canaries;
- private functional evaluation;
- public calibration;
- bulk data ingress;
- live teacher calls;
- dashboard/OpenAI-shim/Hive LAN exposure;
- remote work or arbitrary execution;
- MLX stress tests while presenting.

## Security

Keep every service loopback-only. Do not share credentials, invite files,
private reports, checkpoints, user traces, or ignored local configuration.

Local adversarial tests are not an internet-security audit.

## One-Sentence Explanation

> Theseus is my private local research system for training and comparing a
> useful AI honestly: it preserves exact state, separates learned behavior from
> tools, and refuses to turn infrastructure success into a capability claim.
