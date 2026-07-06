# Deprecated Code And Artifacts

This folder is for source or artifacts that are intentionally retired from the
active runtime but kept briefly for auditability.

Rules:

- Move only code that has current evidence of being unused or superseded.
- Do not move CUDA, benchmark, or safety-gate code merely because it is unused
  in a non-CUDA or narrow local check.
- Add a short note explaining why each item was deprecated and what replaced it.
- Prefer deleting tiny transitional wrappers when the replacement is obvious and
  fully verified.
- Nothing in this folder is part of promotion evidence or the active training
  path.

Current deprecated groups:

- `docs/background/`: retired conceptual/background drafts that are superseded
  by the current docs index, Project State, and generated reports.
- `docs/legacy-transfer/`: retired redirect stubs for old-project transfer
  docs. Use `docs/OLD_PROJECTS_TRANSFER_AUDIT.md` as the canonical page.
- `generated-tmp/`: local-only quarantine for ignored scratch build output
  moved out of the root. Build scripts recreate `tmp/` when needed.
- `logs/legacy-run-logs/`: old tracked runtime logs moved out of the source
  root. New runtime logs are local generated state and ignored by git.
- `windows-drive-mirror/`: local-only quarantine for an accidental Windows
  `D:` path mirror. It is kept for manual reference only and is ignored by git.
