# Project Theseus Public Release Surface

The GitHub repository at `https://github.com/corbensorenson/theseus` is public,
but its current history is **not an approved public-release surface**. The
2026-07-30 maintenance audit found generated evidence and private-evaluation
material in the public tip and history. Freeze pushes while this condition
remains open. A green source-only export is the only candidate publication
surface.

It intentionally does not track local runtime products:

- `reports/`, `runtime/`, `checkpoints/`, `dist/`, `target/`, and logs;
- local personality archives or dogfood traces;
- private generated training rows and private candidate payloads;
- imported benchmark payload clones or benchmark answer files;
- local Hive invite files, secrets, tokens, or machine-specific config.
- deterministic private-evaluation case generators, seeds, verifier details,
  hidden fixtures, consumption ledgers, and exact candidate outputs.

Public benchmarks remain calibration-only. Benchmark prompts, tests, hidden
tests, solutions, traces, and answer templates must not enter training rows.
Generated reports are evidence, not source. Keep them local or publish selected
small summaries through an explicit release artifact.

The current 57M D2 functional surface and its evaluator implementation must not
be copied into a public source tree. The independent readiness audit currently
finds the sealed 160-case surface fresh at exact and whitespace/case-normalized
model-visible prompt scope and verifies its historical consumption identity;
that limited finding does not establish semantic-family independence or
authorize evaluation. The surface, generators, hidden verifiers, seeds,
consumption records, and candidate outputs remain private. Training and D2
remain unavailable until their separate machine-readable exclusive-lease gates
pass.

Before publishing anything, audit the complete tracked repository:

```bash
python3 scripts/public_release_audit.py --gate
```

This command is expected to remain red until the public/private split is
implemented. Do not “fix” it by deleting local evidence before independent
checkpoint custody exists.

Prepare and inspect a source-only working-tree snapshot at a new path outside
the repository:

```bash
python3 scripts/public_release_audit.py \
  --source-release \
  --local-only \
  --prepare-source-tree /private/tmp/theseus-source-release-YYYYMMDD \
  --out /private/tmp/theseus-source-release-audit-YYYYMMDD.json
```

The source-only manifest uses explicit includes, excludes all forbidden
private/evidence paths, rejects secret-like literals, enforces file and total
size caps, writes a SHA-256 inventory, refuses an existing destination, and
marks a dirty-worktree snapshot non-publishable. It does not change Git history,
GitHub visibility, tracking state, or local artifacts.

The current maintenance audit explicitly excludes the licensed P2/P3/P4 task
archives as evidence fixtures rather than source. It finds zero forbidden
selected paths, zero secret-like hard gaps, zero oversized files, and no
protected D2 or evaluator material. Its remaining hard gap is the selected
dirty-worktree set, so it is a review candidate, not a publishable release.
The generated audit owns exact file/byte counts; a prepared tree carries its
content inventory and manifest SHA-256 rather than duplicating a stale digest
here.

## History transition

The selected evidence-safe strategy is `NEW_CLEAN_PUBLIC_ROOT`. Rewriting the
existing public history is not the default because it is destructive, difficult
to audit across forks and caches, and does not undo prior exposure. The complete
current repository becomes a private evidence archive; the public project is
recreated from the audited source-only root.

The transition sequence is:

1. keep pushes frozen;
2. preserve the complete current history in an independently controlled private
   mirror after checkpoint custody is green;
3. treat already-published private/evidence paths as exposed, and rotate any
   credential if a later scan finds one;
4. build a clean root history from a green, clean-worktree source-only export;
5. independently audit that candidate history and its Git object closure;
6. rename or privatize the current repository as the private archive;
7. create a new public repository from the audited clean root, with no parent
   relationship to the unsafe history;
8. verify visibility, object closure, required source files, secrets, licenses,
   and hosted CI before announcing it as the public surface.

Making the existing repository private, creating/replacing a public repository,
renaming it, or rewriting/force-pushing history are external or destructive
actions and require explicit operator approval. The selected strategy does not
authorize those actions; no cleanup command may silently perform them. A
force-push rewrite is retained only as an operator-requested exception with its
own object-closure audit and rollback custody.

The audit verifies GitHub visibility when `require_public_visibility` is enabled
in `configs/public_release_manifest.json`. Visibility being public is necessary
for a final book reference, but it is not evidence that the current history is
safe.

The manifest also owns the public root allowlist. New top-level tracked files
or directories should be added only when they are intentional public source
surfaces. Otherwise they should live under an existing source root, stay ignored
as local runtime state, or be moved to `deprecated/` with a short rationale.
