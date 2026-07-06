# Theseus Compute Market

The Theseus Compute Market is the accounting layer for rented Hive compute.
It lets a weak client quote a stronger node before sending work, lets a worker
return a signed-style work receipt after bounded execution, and lets the local
wallet record internal Theseus Work Credits.

This is **not** a live crypto exchange yet. The current implementation is
internal accounting only:

- no fiat on-ramp;
- no custody;
- no public order book;
- no tradable public token;
- no exchange matching;
- no arbitrary public tasks.

Those pieces require a separate legal, security, anti-abuse, tax, custody, and
public-chain release. The local code reserves the public-chain symbol idea but
keeps the actual unit as an internal `TWC` meter.

## Flow

```text
client wants work
  -> scheduler selects candidate Hive worker
  -> compute market quotes gas
  -> bounded task runs on worker
  -> worker writes work receipt
  -> settlement verifies receipt and task kind
  -> provider wallet receives internal work credits
  -> ledger records gas, payout, fees, and receipt id
```

## Files

| File | Purpose |
| --- | --- |
| `configs/compute_market_policy.json` | Currency, pricing, gas, settlement, rental, and legal-posture policy. |
| `scripts/compute_market.py` | CLI/API module for status, quote, rent-plan, and receipt settlement. |
| `reports/compute_market_status.json` | Current wallet, balances, legal posture, and ledger summary. |
| `reports/compute_market_quote_last.json` | Last gas quote. |
| `reports/compute_market_settlement_last.json` | Last settlement batch or receipt result. |
| `reports/compute_market_ledger.jsonl` | Append-only internal accounting events. |
| `reports/compute_market_receipts.jsonl` | Accepted work receipts with settlement metadata. |
| `configs/compute_market_wallet.local.json` | Local ignored wallet/account state. |

`reports/` and `configs/*.local.json` are ignored, so wallet state and local
receipts do not enter the public repository.

## CLI

```powershell
theseus market status
theseus market quote --task-kind cuda_eval_chunk --payload-json "{\"profile\":\"smoke\",\"cases_per_task\":4,\"epochs\":1,\"samples_per_launch\":64,\"hv_dim\":512}"
theseus market rent-plan --task-kind cuda_eval_chunk --payload-json "{\"profile\":\"smoke\"}"
theseus market settle --worker-ledger reports/hive_worker_chunk_ledger.jsonl
```

The dashboard exposes the same controls in the `Compute Market` panel:

- `Check` refreshes the local wallet and ledger view;
- `Quote CUDA` creates a sample quote against the local CUDA-style task;
- `Settle Receipts` scans recent Hive worker-chunk receipts and records any
  unsettled accepted work.

## Pricing

The quote function uses:

```text
work units
  * task-kind base rate
  * backend multiplier
  * difficulty/profile multiplier
  -> gas estimate
```

The current unit is `micro_twc`, where:

```text
1 TWC = 1,000,000 micro_twc
```

Rates are deliberately small and accounting-only while the project is local.
The policy supports separate rates for CUDA, MLX, CPU, chat gateways, public
smoke tasks, public eval shards, and synthetic-data quality votes.

## Receipts

Worker chunks already emit `work_receipt` blocks. A receipt contains:

- task kind;
- backend;
- profile/difficulty;
- claimed work units;
- runtime;
- verifier label;
- acceptance bit;
- anti-cheat status;
- provider account when known.

`scripts/compute_market.py settle` rejects duplicate receipts, rejected work,
unknown task kinds, and zero-unit claims. Public rewards will eventually need
stronger anti-cheat than the local private-Hive receipts.

## Public Hive Path

The intended public path is:

```text
signed public task manifest
  -> sandboxed worker execution
  -> deterministic replay or spot check
  -> result consensus or holdout verification
  -> receipt settlement
  -> public-chain token bridge
```

Until those gates exist, public contribution remains worker-only and
accounting-only. The exchange/token switches in
`configs/compute_market_policy.json` stay `false`.

## Legal Posture

Crypto and exchange operation can trigger securities, commodities,
money-transmission, tax, sanctions, and consumer-protection obligations. The
code therefore separates:

- internal work metering, which is implemented now;
- public gateway operation, which is licensed and gated;
- public-chain token issuance, which is disabled;
- exchange/custody/on-ramp operation, which is disabled.

Before enabling a public token or exchange, this project needs a legal review
and a production security design.
