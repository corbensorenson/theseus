# Project Theseus Licensing System

The Project Theseus licensing layer is a local-first product gate for the app,
Hive, distributed worker chunks, compute-market accounting, company hives, and
public-network operation.
It is designed to stop accidental unlicensed growth early while keeping the
private license-issuing key outside the repository.

## What It Enforces

The active policy lives in `configs/license_policy.json`.

| Scope | Default behavior |
| --- | --- |
| Local status checks | Allowed without registration. |
| Local research, private Hive use, and update install | Requires local registration and terms acceptance. |
| Free community use | Non-commercial use up to 12 nodes/users. |
| Distributed CUDA/MLX worker chunks | Requires registration or a signed paid license. |
| Internal compute-market quotes/accounting | Requires registration; exchange/token operation is disabled separately. |
| Company Hive and commercial use | Requires signed paid license. |
| Public Hive gateway operation | Requires signed paid `public_operator` or `enterprise` license. |
| Private/company update channel | Requires signed `enterprise` license; local/company arms remain protected. |

Community registration is for personal homelab, research, friends/family, and
small startup-style non-commercial use. It is not a commercial-company license.

## Local Files

| File | Purpose | Git status |
| --- | --- | --- |
| `configs/license_policy.json` | Versioned licensing policy, tiers, gates, and public-key slots. | Tracked |
| `configs/theseus_registration.local.json` | Local install registration and accepted terms. | Ignored |
| `configs/theseus_license.local.json` | Imported signed paid license. | Ignored |
| `reports/license_status.json` | Current gate/status report. | Ignored |
| `reports/license_request.json` | Bundle to request a paid license. | Ignored |
| `reports/license_events.jsonl` | Local registration/import/request event trail. | Ignored |

## CLI

Register a non-commercial local install:

```powershell
py scripts\license_manager.py register --name "Your Name" --usage personal_homelab --seats 1 --accept-terms
```

Check status:

```powershell
py scripts\license_manager.py status
theseus license status
```

Create a paid license request bundle:

```powershell
py scripts\license_manager.py request --feature company_hive --feature multi_network_company_relay
```

Import a signed paid license:

```powershell
py scripts\license_manager.py import --file path\to\signed-license.json
```

Check a specific feature gate:

```powershell
py scripts\license_manager.py check --feature distributed_worker_chunks
py scripts\license_manager.py check --feature compute_market_accounting
py scripts\license_manager.py check --feature compute_rental_client
py scripts\license_manager.py check --feature public_work_accounting
py scripts\license_manager.py check --feature update_install
py scripts\license_manager.py check --feature private_update_channel
py scripts\license_manager.py check --feature company_hive --requested-tier company
```

The `theseus` CLI exposes the same operations through `theseus license ...`.

## Dashboard And Setup Wizard

The SparkStream dashboard exposes the current registration/license status in
the hero metrics and the "Registration & License" panel. The setup wizard also
has registration, request, and import actions for non-terminal installs.

The dashboard and Hive APIs do not bypass licensing. Hive profile creation,
invite creation/application, task submission, worker-chunk scheduling, launch
readiness, and public contribution reports all read the same gate state.

## Signed Paid Licenses

Paid licenses are JSON payloads signed with Ed25519. The canonical payload is
the license JSON without the `signature` field, serialized with sorted keys and
compact separators. Public verification keys are listed in
`configs/license_policy.json` under `issuer.signature_public_keys`.

The private issuing key must never be committed to this repository. A release
build should include only public verification keys.

## Release Note

The runtime licensing layer is a product-control and safety gate. If the source
code is distributed under a permissive open-source license, recipients can
remove local checks from their copy. Before a broad public/company release,
choose the final business license terms deliberately and make sure they match
the intended commercial model.
