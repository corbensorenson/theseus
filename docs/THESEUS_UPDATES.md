# Project Theseus Candidate Updates

Project Theseus updates are tied to accepted candidates, not ordinary
experiments. A candidate becomes installable only after the promotion gate
passes and a promoted checkpoint exists.

## Update Flow

```text
candidate gate promotes
    -> checkpoint registry creates promoted checkpoint
    -> checkpoint backup manager emits accepted-candidate manifest
    -> update manager creates update offer
    -> dashboard / CLI / Hive nodes announce update availability
    -> soft update installs checkpoint metadata immediately
    -> hard update stages app/source changes and requires explicit restart
```

The main files are:

| File | Role |
| --- | --- |
| `configs/update_policy.json` | Versioned update policy, channels, protected paths, soft/hard rules. |
| `scripts/update_manager.py` | Creates offers, reports status, applies soft updates, guards hard updates. |
| `reports/update_offer_current.json` | Current accepted-candidate update offer. |
| `reports/update_status.json` | Machine-readable update status for dashboard, CLI, and Hive. |
| `configs/update_installed.local.json` | Local installed update state, ignored by git. |
| `updates/` | Ignored staging/materialization/install-manifest directory. |

## Soft Versus Hard Updates

Soft updates activate candidate/checkpoint metadata and tell the user what the
candidate is better at. They do not replace source code, app files, local data,
ROMs, reports, or learned local/company arms, and they do not require restart.

Hard updates are used when the accepted checkpoint includes source, dashboard,
Rust crate, config, docs, adapter, or benchmark-card changes. The hard path
materializes the checkpoint under `updates/materialized/`, copies only allowed
workspace files, skips protected paths, and reports `restart_required=true`.

Hard updates require both:

```powershell
py scripts\update_manager.py apply --mode hard --execute --allow-hard
```

The dashboard intentionally exposes soft install as the easy path. Hard updates
are a guarded operator action because they can replace app/source files.

## Protected Company And Local Assets

The update policy always protects:

- `configs/*.local.json` and `configs/*.secret.json`
- registration/license files
- Hive join/profile files
- reports, checkpoints, updates, data, games, target, environments
- local/company arms under `arms/local/**` and `arms/company/**`
- ROMs, emulator media, model binaries, private keys, and environment files

Arms can also be protected by local config:

```powershell
py scripts\update_manager.py protect-arm my_company_arm --reason "customer learned arm"
```

This writes `configs/update_client.local.json`, which is ignored by git.

## CLI

```powershell
py scripts\update_manager.py status --out reports/update_status.json
py scripts\update_manager.py create --if-promoted --out reports/update_offer_current.json
py scripts\update_manager.py apply --mode soft --execute --out reports/update_apply_last.json
```

Through the installed CLI:

```powershell
theseus update status
theseus update create --if-promoted
theseus update apply --mode soft --execute
```

## Dashboard And Hive

The dashboard includes a Candidate Updates panel and exposes:

- `POST /api/updates/status`
- `POST /api/updates/create`
- `POST /api/updates/apply`

Hive nodes advertise `candidate_update_client` and can run bounded
`update_status` and `update_apply_soft` tasks. Hard updates, arbitrary shell,
teacher calls, git pushes, ROM imports, and bulk downloads remain forbidden as
remote task kinds.

## Licensing

Community installs can use `update_install` after local registration. Enterprise
licenses can use `private_update_channel` for private/company update channels.
Company-local learned arms remain protected even when update installation is
licensed.
