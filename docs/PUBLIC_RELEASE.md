# Project Theseus Public Release Surface

This repository is public at
`https://github.com/corbensorenson/symliquid-rmi`. It is intended to publish
the source, configuration, docs, apps, schemas, and small reproducible
manifests for Project Theseus so *The ASI Stack* book can link to a stable,
auditable implementation reference.

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

The audit also verifies GitHub visibility when `require_public_visibility` is
enabled in `configs/public_release_manifest.json`. A private GitHub repository
is a hard release gap because the book cannot safely treat it as a public
reference.

The manifest also owns the public root allowlist. New top-level tracked files
or directories should be added only when they are intentional public source
surfaces. Otherwise they should live under an existing source root, stay ignored
as local runtime state, or be moved to `deprecated/` with a short rationale.
