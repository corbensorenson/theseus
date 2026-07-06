# Checkpoint Backups

Project Theseus checkpoints are local major/minor chains. Accepted candidate
backups are a narrower off-machine safety lane for moments when the candidate
promotion gate actually passes.

## Rule

Do not back up every experiment as an accepted candidate.

The backup manager only treats a checkpoint as accepted when:

- `reports/candidate_promotion_gate.json` has `promote=true`;
- the checkpoint status is `promoted`;
- the checkpoint manifest can be resolved;
- the backup manifest does not reference forbidden paths.

Current command:

```powershell
py -3.13 scripts\checkpoint_backup_manager.py --if-promoted --execute --provider all
```

When `promote=false`, the manager writes `skipped_not_promoted` and performs no
GitHub push or Drive queue upload.

Accepted-candidate backup also emits an installable update offer through
`scripts/update_manager.py`. The update offer is local metadata by default; it
does not publish datasets, ROMs, generated reports, model binaries, or
materialized workspace contents.

## Providers

GitHub is the default code/manifest backup provider. It backs up:

- tracked source already in git;
- a small accepted-candidate manifest under
  `backup_manifests/accepted_candidates/`;
- the current branch, pushed to the configured remote.

Google Drive is queue-only by default. The local Python runner cannot directly
call the Codex Google Drive connector, so it writes a queue item to:

```text
reports/google_drive_backup_queue.jsonl
```

A future app connector worker can upload those queued manifests and small
metadata bundles.

## Forbidden Scope

The backup lane does not include:

- ROMs or game archives;
- generated reports;
- local training data;
- external benchmark candidates;
- synthetic data;
- target build artifacts;
- materialized checkpoint workspaces;
- model binary formats such as `.pt`, `.safetensors`, `.onnx`, or `.bin`.

That keeps private/local assets local and avoids accidentally publishing data
whose license or size does not belong in GitHub.

## Autonomy Integration

Each autonomy cycle creates a checkpoint as before. Immediately after that, it
runs:

```text
scripts/checkpoint_backup_manager.py --if-promoted ...
```

This makes accepted-candidate backup automatic without turning every ordinary
inner-loop checkpoint into an off-machine artifact.

If the candidate promotes, the same cycle also creates
`reports/update_offer_current.json` and attempts the configured automatic soft
install. Hard app/source updates are staged and restart-gated; protected
local/company arms and local configs are skipped.

Machine-readable reports:

- `reports/checkpoint_backup_last.json`
- `reports/checkpoint_backup_history.jsonl`
- `reports/checkpoint_registry.json`
- `reports/update_offer_current.json`
- `reports/update_status.json`
