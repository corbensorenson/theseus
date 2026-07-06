# Project Theseus Public Release Surface

This repository is intended to publish the source, configuration, docs, apps,
schemas, and small reproducible manifests for Project Theseus.

It intentionally does not track local runtime products:

- `reports/`, `runtime/`, `checkpoints/`, `dist/`, `target/`, and logs;
- local personality archives or dogfood traces;
- private generated training rows and private candidate payloads;
- imported benchmark payload clones or benchmark answer files;
- local Hive invite files, secrets, tokens, or machine-specific config.

Public benchmarks remain calibration-only. Benchmark prompts, tests, hidden
tests, solutions, traces, and answer templates must not enter training rows.
Generated reports are evidence, not source. Keep them local or publish selected
small summaries through an explicit release artifact.

Before changing GitHub visibility or publishing a release snapshot, run:

```bash
python3 scripts/public_release_audit.py --gate
```

The audit is intentionally conservative. If it reports a forbidden tracked path,
remove that path from git tracking and keep it in local storage or a separate
private artifact store.
