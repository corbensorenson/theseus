"""Internal compute-market accounting for Project Theseus Hive.

This module is intentionally an accounting layer, not a live exchange. It
quotes gas for bounded Hive work, records verified work receipts, settles
internal Theseus Work Credits, and exposes enough state for the scheduler,
dashboard, public contribution bridge, and CLI to reason about rented compute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "compute_market_policy.json"
LICENSE_POLICY_PATH = ROOT / "configs" / "license_policy.json"
WORKER_KIND_TASK_ALIASES = {
    "cuda_readout_train": "cuda_training_chunk",
    "cuda_rollout_train": "cuda_rollout_chunk",
    "mlx_babylm_eval": "mlx_eval_chunk",
    "mlx_babylm_train": "mlx_training_chunk",
    "mlx_rollout_probe": "mlx_rollout_chunk",
}

sys.path.insert(0, str(ROOT / "scripts"))
import license_manager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(POLICY_PATH.relative_to(ROOT)))
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status")
    status.add_argument("--out", default="")

    quote = sub.add_parser("quote")
    quote.add_argument("--task-kind", required=True)
    quote.add_argument("--payload-json", default="{}")
    quote.add_argument("--provider-node-json", default="{}")
    quote.add_argument("--out", default="")

    settle = sub.add_parser("settle")
    settle.add_argument("--receipt-json", default="")
    settle.add_argument("--receipt-file", default="")
    settle.add_argument("--consumer-account", default="")
    settle.add_argument("--provider-account", default="")
    settle.add_argument("--out", default="")

    settle_ledger_cmd = sub.add_parser("settle-ledger")
    settle_ledger_cmd.add_argument("--worker-ledger", default="reports/hive_worker_chunk_ledger.jsonl")
    settle_ledger_cmd.add_argument("--limit", type=int, default=50)
    settle_ledger_cmd.add_argument("--out", default="")

    rent = sub.add_parser("rent-plan")
    rent.add_argument("--task-kind", required=True)
    rent.add_argument("--payload-json", default="{}")
    rent.add_argument("--max-gas-micro-twc", type=int, default=0)
    rent.add_argument("--out", default="")

    args = parser.parse_args()
    policy = read_json(ROOT / args.policy, {})
    if args.command in {None, "status"}:
        report = status_report(policy=policy, write_report=True)
        out = getattr(args, "out", "") or ""
    elif args.command == "quote":
        payload = parse_json(args.payload_json, {})
        provider = parse_json(args.provider_node_json, {})
        report = quote_task(args.task_kind, payload, provider, policy=policy, write_report=True)
        out = args.out
    elif args.command == "settle":
        receipt = load_receipt_arg(args)
        context = {
            "consumer_account": args.consumer_account,
            "provider_account": args.provider_account,
            "source": "cli",
        }
        report = settle_receipt(receipt, context=context, policy=policy, write_report=True)
        out = args.out
    elif args.command == "settle-ledger":
        report = settle_worker_ledger(ROOT / args.worker_ledger, limit=args.limit, policy=policy, write_report=True)
        out = args.out
    elif args.command == "rent-plan":
        payload = parse_json(args.payload_json, {})
        report = rent_plan(args.task_kind, payload, max_gas_micro_twc=args.max_gas_micro_twc, policy=policy, write_report=True)
        out = args.out
    else:
        parser.print_help()
        return 2
    if out:
        write_json(ROOT / out, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) else 2


def status_report(*, policy: dict[str, Any] | None = None, write_report: bool = False) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    wallet = load_wallet(policy)
    ledger_tail = read_jsonl_tail(path_from_policy(policy, ["paths", "ledger"], "reports/compute_market_ledger.jsonl"), 100)
    receipt_tail = read_jsonl_tail(path_from_policy(policy, ["paths", "receipts"], "reports/compute_market_receipts.jsonl"), 100)
    license_status = license_manager.status_report(write_report=True)
    accounting_license = license_manager.check_feature("public_work_accounting", policy=license_manager.read_json(LICENSE_POLICY_PATH, {}), write_report=True)
    summary = summarize_ledger(ledger_tail)
    report = {
        "ok": True,
        "policy": "project_theseus_compute_market_status_v0",
        "created_utc": now(),
        "enabled": bool(policy.get("enabled", True)),
        "mode": get_path(policy, ["legal_posture", "mode"], "internal_accounting_only"),
        "tradable_token_enabled": bool(get_path(policy, ["legal_posture", "tradable_token_enabled"], False)),
        "exchange_enabled": bool(get_path(policy, ["legal_posture", "exchange_enabled"], False)),
        "currency": policy.get("currency", {}),
        "wallet": public_wallet(wallet),
        "balances": wallet.get("balances", {}),
        "summary": summary,
        "last_quote": read_json(path_from_policy(policy, ["paths", "last_quote"], "reports/compute_market_quote_last.json"), {}),
        "recent_receipts": receipt_tail[-12:],
        "recent_ledger": ledger_tail[-12:],
        "license": {
            "tier": get_path(license_status, ["entitlement", "tier"], ""),
            "source": get_path(license_status, ["entitlement", "source"], ""),
            "can_account_public_work": bool(accounting_license.get("allowed")),
            "next_action": accounting_license.get("next_action"),
        },
        "risk_controls": policy.get("risk_controls", {}),
        "next_action": next_action(policy, wallet, accounting_license),
        "external_inference_calls": 0,
    }
    if write_report:
        write_json(path_from_policy(policy, ["paths", "status"], "reports/compute_market_status.json"), report)
    return report


def quote_task(
    task_kind: str,
    payload: dict[str, Any] | None = None,
    provider_node: dict[str, Any] | None = None,
    *,
    policy: dict[str, Any] | None = None,
    write_report: bool = False,
) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    payload = payload or {}
    provider_node = provider_node or {}
    units = estimate_work_units(task_kind, payload, policy)
    backend = infer_backend(provider_node, payload)
    profile = str(payload.get("profile") or "smoke")
    base_rate = int(get_path(policy, ["pricing", "task_kind_rates", task_kind], get_path(policy, ["pricing", "default_base_micro_twc_per_million_work_units"], 12)))
    backend_multiplier = float(get_path(policy, ["pricing", "backend_multipliers", backend], 1.0))
    difficulty_multiplier = float(get_path(policy, ["pricing", "difficulty_multipliers", profile], 1.0))
    minimum = int(get_path(policy, ["pricing", "minimum_gas_micro_twc"], 1))
    raw_gas = units / 1_000_000.0 * base_rate * backend_multiplier * difficulty_multiplier
    gas = max(minimum, int(round(raw_gas)))
    fee_bps = int(get_path(policy, ["pricing", "protocol_fee_bps"], 200))
    protocol_fee = max(0, int(round(gas * fee_bps / 10_000.0)))
    provider_payout = max(0, gas - protocol_fee)
    quote = {
        "ok": True,
        "policy": "project_theseus_compute_market_quote_v0",
        "created_utc": now(),
        "task_kind": task_kind,
        "profile": profile,
        "backend": backend,
        "estimated_work_units": units,
        "difficulty_class": profile,
        "currency_symbol": get_path(policy, ["currency", "symbol"], "TWC"),
        "gas_estimate_micro_twc": gas,
        "provider_payout_micro_twc": provider_payout,
        "protocol_fee_micro_twc": protocol_fee,
        "base_rate_micro_twc_per_million_work_units": base_rate,
        "backend_multiplier": backend_multiplier,
        "difficulty_multiplier": difficulty_multiplier,
        "accounting_only": get_path(policy, ["legal_posture", "mode"], "") == "internal_accounting_only",
        "tradable_token_enabled": bool(get_path(policy, ["legal_posture", "tradable_token_enabled"], False)),
        "provider_node": compact_provider(provider_node),
        "external_inference_calls": 0,
    }
    quote["quote_id"] = stable_id("quote", quote)
    if write_report:
        write_json(path_from_policy(policy, ["paths", "last_quote"], "reports/compute_market_quote_last.json"), quote)
    return quote


def rent_plan(
    task_kind: str,
    payload: dict[str, Any],
    *,
    max_gas_micro_twc: int = 0,
    policy: dict[str, Any] | None = None,
    write_report: bool = False,
) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    wallet = load_wallet(policy)
    quote = quote_task(task_kind, payload, {}, policy=policy, write_report=True)
    max_gas = max_gas_micro_twc or int(get_path(policy, ["rental", "default_max_gas_micro_twc"], 100000))
    balance = int(get_path(wallet, ["balances", "available_micro_twc"], 0))
    allow_negative = bool(get_path(policy, ["rental", "allow_negative_local_test_balance"], True))
    can_afford = balance >= int(quote.get("gas_estimate_micro_twc") or 0) or allow_negative
    within_budget = int(quote.get("gas_estimate_micro_twc") or 0) <= max_gas
    report = {
        "ok": True,
        "policy": "project_theseus_compute_market_rent_plan_v0",
        "created_utc": now(),
        "task_kind": task_kind,
        "quote": quote,
        "wallet": public_wallet(wallet),
        "max_gas_micro_twc": max_gas,
        "can_afford": can_afford,
        "within_budget": within_budget,
        "route_decision": "rent_from_hive" if can_afford and within_budget else "do_not_rent",
        "reason": "quote accepted" if can_afford and within_budget else "insufficient gas budget or wallet balance",
        "rental_policy": policy.get("rental", {}),
        "external_inference_calls": 0,
    }
    if write_report:
        write_json(path_from_policy(policy, ["paths", "rent_plan"], "reports/compute_market_rent_plan.json"), report)
    return report


def settle_receipt(
    receipt: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
    write_report: bool = False,
) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    context = context or {}
    valid = verify_receipt(receipt, policy)
    if not valid.get("ok"):
        report = {
            "ok": False,
            "policy": "project_theseus_compute_market_settlement_v0",
            "created_utc": now(),
            "error": valid.get("error"),
            "verification": valid,
            "receipt": receipt,
            "external_inference_calls": 0,
        }
        if write_report:
            write_json(path_from_policy(policy, ["paths", "last_settlement"], "reports/compute_market_settlement_last.json"), report)
        return report

    receipt_id = receipt_hash(receipt)
    ledger_path = path_from_policy(policy, ["paths", "ledger"], "reports/compute_market_ledger.jsonl")
    if receipt_seen(ledger_path, receipt_id):
        report = {
            "ok": False,
            "policy": "project_theseus_compute_market_settlement_v0",
            "created_utc": now(),
            "error": "duplicate_receipt",
            "receipt_id": receipt_id,
            "external_inference_calls": 0,
        }
        if write_report:
            write_json(path_from_policy(policy, ["paths", "last_settlement"], "reports/compute_market_settlement_last.json"), report)
        return report

    task_kind = receipt_task_kind(receipt) or "unknown"
    quote = quote_task(
        task_kind,
        {
            "profile": receipt.get("profile") or "smoke",
            "claimed_work_units": receipt.get("claimed_work_units"),
        },
        {"capabilities": [{"id": str(receipt.get("backend") or "cpu")}]},
        policy=policy,
        write_report=False,
    )
    actual_units = clamp_int(receipt.get("claimed_work_units"), 1, int(get_path(policy, ["work_units", "max_claimed_units_per_receipt"], 100_000_000_000)))
    quoted_units = max(1, int(quote.get("estimated_work_units") or actual_units))
    scale = actual_units / quoted_units
    gas = max(int(get_path(policy, ["pricing", "minimum_gas_micro_twc"], 1)), int(round(int(quote["gas_estimate_micro_twc"]) * scale)))
    fee_bps = int(get_path(policy, ["pricing", "protocol_fee_bps"], 200))
    fee = max(0, int(round(gas * fee_bps / 10_000.0)))
    payout = max(0, gas - fee)
    wallet = load_wallet(policy)
    provider_account = str(context.get("provider_account") or receipt.get("provider_account") or wallet.get("account_id") or "")
    consumer_account = str(context.get("consumer_account") or receipt.get("consumer_account") or "protocol_reward_pool")
    event = {
        "policy": "project_theseus_compute_market_ledger_event_v0",
        "event_id": f"market_evt_{uuid.uuid4().hex}",
        "created_utc": now(),
        "event": "settle_work_receipt",
        "receipt_id": receipt_id,
        "task_kind": task_kind,
        "worker_kind": receipt.get("worker_kind") or receipt.get("kind") or task_kind,
        "provider_account": provider_account,
        "consumer_account": consumer_account,
        "work_units": actual_units,
        "gas_micro_twc": gas,
        "provider_payout_micro_twc": payout,
        "protocol_fee_micro_twc": fee,
        "accounting_only": True,
        "source": context.get("source") or "local",
        "quote": {key: quote.get(key) for key in ["quote_id", "backend", "difficulty_class", "base_rate_micro_twc_per_million_work_units"]},
    }
    append_jsonl(ledger_path, event)
    receipt_record = {**receipt, "receipt_id": receipt_id, "settled_utc": now(), "settlement_event_id": event["event_id"]}
    append_jsonl(path_from_policy(policy, ["paths", "receipts"], "reports/compute_market_receipts.jsonl"), receipt_record)
    apply_wallet_settlement(policy, wallet, payout, fee, receipt_id)
    report = {
        "ok": True,
        "policy": "project_theseus_compute_market_settlement_v0",
        "created_utc": now(),
        "receipt_id": receipt_id,
        "event": event,
        "wallet": public_wallet(wallet),
        "external_inference_calls": 0,
    }
    if write_report:
        write_json(path_from_policy(policy, ["paths", "last_settlement"], "reports/compute_market_settlement_last.json"), report)
        status_report(policy=policy, write_report=True)
    return report


def settle_worker_ledger(
    worker_ledger: Path,
    *,
    limit: int,
    policy: dict[str, Any] | None = None,
    write_report: bool = False,
) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    rows = read_jsonl_tail(worker_ledger, max(1, limit))
    settlements = []
    for row in rows:
        receipt = row.get("work_receipt") if isinstance(row, dict) and isinstance(row.get("work_receipt"), dict) else {}
        if not receipt:
            continue
        settlements.append(settle_receipt(receipt, context={"source": str(worker_ledger.relative_to(ROOT)) if worker_ledger.is_relative_to(ROOT) else str(worker_ledger)}, policy=policy, write_report=False))
    report = {
        "ok": True,
        "policy": "project_theseus_compute_market_settle_worker_ledger_v0",
        "created_utc": now(),
        "worker_ledger": str(worker_ledger),
        "scanned": len(rows),
        "settled": sum(1 for row in settlements if row.get("ok")),
        "duplicates_or_rejected": sum(1 for row in settlements if not row.get("ok")),
        "settlements": settlements[-20:],
        "external_inference_calls": 0,
    }
    if write_report:
        write_json(path_from_policy(policy, ["paths", "last_settlement"], "reports/compute_market_settlement_last.json"), report)
        status_report(policy=policy, write_report=True)
    return report


def verify_receipt(receipt: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        return {"ok": False, "error": "receipt_must_be_object"}
    if bool(get_path(policy, ["settlement", "require_receipt_accepted"], True)) and not bool(receipt.get("accepted")):
        return {"ok": False, "error": "receipt_not_accepted"}
    task_kind = receipt_task_kind(receipt)
    if not task_kind:
        return {"ok": False, "error": "task_kind_required"}
    allowed = set(get_path(read_json(ROOT / "configs" / "hive_policy.json", {}), ["task_kinds"], {}).keys())
    if get_path(policy, ["settlement", "require_registered_task_kind"], True) and task_kind not in allowed:
        return {"ok": False, "error": "unregistered_task_kind", "task_kind": task_kind}
    units = clamp_int(receipt.get("claimed_work_units"), 0, int(get_path(policy, ["work_units", "max_claimed_units_per_receipt"], 100_000_000_000)))
    if units <= 0:
        return {"ok": False, "error": "work_units_required"}
    return {"ok": True, "task_kind": task_kind, "claimed_work_units": units}


def receipt_task_kind(receipt: dict[str, Any]) -> str:
    """Normalize current and pre-market worker receipt schemas."""
    task_kind = str(receipt.get("task_kind") or "")
    if task_kind:
        return task_kind
    worker_kind = str(receipt.get("worker_kind") or receipt.get("kind") or "")
    return WORKER_KIND_TASK_ALIASES.get(worker_kind, worker_kind)


def estimate_work_units(task_kind: str, payload: dict[str, Any], policy: dict[str, Any]) -> int:
    if payload.get("claimed_work_units") is not None:
        return clamp_int(payload.get("claimed_work_units"), 1, int(get_path(policy, ["work_units", "max_claimed_units_per_receipt"], 100_000_000_000)))
    defaults = get_path(policy, ["work_units", "payload_fields"], {})
    cases = value_int(payload, "cases_per_task", int(defaults.get("cases_per_task", 4)))
    epochs = max(value_int(payload, "epochs", int(defaults.get("epochs", 1))), value_int(payload, "steps", int(defaults.get("steps", 1))))
    samples = value_int(payload, "samples_per_launch", int(defaults.get("samples_per_launch", 64)))
    hv_dim = value_int(payload, "hv_dim", value_int(payload, "feature_dim", int(defaults.get("hv_dim", 512))))
    train = value_int(payload, "train_limit", int(defaults.get("train_limit", 128)))
    eval_rows = value_int(payload, "eval_limit", int(defaults.get("eval_limit", 128)))
    rollout = value_int(payload, "rollout_batch", int(defaults.get("rollout_batch", 4)))
    seq_len = value_int(payload, "seq_len", int(defaults.get("seq_len", 8)))
    if task_kind.startswith("cuda_rollout"):
        units = max(1, cases) * max(1, epochs) * max(1, rollout) * max(1, seq_len) * max(1, hv_dim)
    elif task_kind.startswith("cuda") or task_kind in {"training_smoke"}:
        units = max(1, cases) * max(1, epochs) * max(1, samples) * max(1, hv_dim)
    elif task_kind.startswith("mlx"):
        units = max(1, train + eval_rows) * max(1, epochs) * max(1, hv_dim)
    elif task_kind == "checkpoint_chat":
        units = max(1, len(str(payload.get("prompt") or "")) * 1024)
    else:
        units = int(get_path(policy, ["work_units", "default_units"], 1_000_000))
    return clamp_int(units, 1, int(get_path(policy, ["work_units", "max_claimed_units_per_receipt"], 100_000_000_000)))


def infer_backend(provider_node: dict[str, Any], payload: dict[str, Any]) -> str:
    if payload.get("backend"):
        return str(payload.get("backend"))
    capabilities = provider_node.get("capabilities") if isinstance(provider_node.get("capabilities"), list) else []
    ids = {str(cap.get("id") or "") for cap in capabilities if isinstance(cap, dict)}
    if "nvidia_cuda" in ids or "rust_cuda" in ids:
        return "nvidia_cuda"
    if "apple_mlx" in ids or "mlx_apple" in ids:
        return "mlx_apple"
    if "mlx_cuda" in ids:
        return "mlx_cuda"
    if "checkpoint_chat_gateway" in ids:
        return "checkpoint_chat_gateway"
    return "cpu"


def apply_wallet_settlement(policy: dict[str, Any], wallet: dict[str, Any], payout: int, fee: int, receipt_id: str) -> None:
    balances = wallet.setdefault("balances", {})
    balances["earned_micro_twc"] = int(balances.get("earned_micro_twc") or 0) + payout
    balances["available_micro_twc"] = int(balances.get("available_micro_twc") or 0) + payout
    balances["protocol_fee_paid_micro_twc"] = int(balances.get("protocol_fee_paid_micro_twc") or 0) + fee
    wallet["updated_utc"] = now()
    wallet["last_receipt_id"] = receipt_id
    write_json(path_from_policy(policy, ["paths", "wallet"], "configs/compute_market_wallet.local.json"), wallet)


def load_wallet(policy: dict[str, Any]) -> dict[str, Any]:
    path = path_from_policy(policy, ["paths", "wallet"], "configs/compute_market_wallet.local.json")
    wallet = read_json(path, {})
    if isinstance(wallet, dict) and wallet.get("account_id"):
        return wallet
    identity = read_json(ROOT / "reports" / "hive_node_identity.json", {})
    seed = "|".join([str(identity.get("node_id") or ""), socket.gethostname(), platform.system(), str(ROOT)])
    account_id = "twc_" + hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()[:32]
    wallet = {
        "policy": "project_theseus_compute_market_wallet_v0",
        "created_utc": now(),
        "account_id": account_id,
        "node_id": identity.get("node_id") or "",
        "node_name": identity.get("node_name") or socket.gethostname(),
        "custodial": False,
        "local_only": True,
        "balances": {
            "available_micro_twc": 0,
            "earned_micro_twc": 0,
            "spent_micro_twc": 0,
            "protocol_fee_paid_micro_twc": 0
        }
    }
    write_json(path, wallet)
    return wallet


def public_wallet(wallet: dict[str, Any]) -> dict[str, Any]:
    return {
        "account_id": wallet.get("account_id"),
        "node_id": wallet.get("node_id"),
        "node_name": wallet.get("node_name"),
        "custodial": bool(wallet.get("custodial")),
        "local_only": bool(wallet.get("local_only", True)),
        "updated_utc": wallet.get("updated_utc") or wallet.get("created_utc"),
    }


def summarize_ledger(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gas = sum(int(row.get("gas_micro_twc") or 0) for row in rows)
    payout = sum(int(row.get("provider_payout_micro_twc") or 0) for row in rows)
    fees = sum(int(row.get("protocol_fee_micro_twc") or 0) for row in rows)
    by_task: dict[str, int] = {}
    for row in rows:
        task = str(row.get("task_kind") or "unknown")
        by_task[task] = by_task.get(task, 0) + 1
    return {
        "ledger_tail_events": len(rows),
        "gas_micro_twc_tail": gas,
        "provider_payout_micro_twc_tail": payout,
        "protocol_fee_micro_twc_tail": fees,
        "by_task_tail": by_task,
    }


def next_action(policy: dict[str, Any], wallet: dict[str, Any], accounting_license: dict[str, Any]) -> str:
    if not policy.get("enabled", True):
        return "Compute market accounting is disabled by policy."
    if get_path(policy, ["legal_posture", "exchange_enabled"], False):
        return "Exchange mode is enabled; verify legal/compliance controls before public operation."
    if not accounting_license.get("allowed"):
        return "Internal accounting works locally; public work accounting requires registration or a public-operator license."
    if int(get_path(wallet, ["balances", "earned_micro_twc"], 0)) <= 0:
        return "Run a bounded worker chunk or public work smoke to generate the first work receipt."
    return "Accounting is active. Use quotes before renting compute and settle receipts after verified work."


def load_receipt_arg(args: argparse.Namespace) -> dict[str, Any]:
    if args.receipt_json:
        return parse_json(args.receipt_json, {})
    if args.receipt_file:
        return read_json((ROOT / args.receipt_file).resolve() if not Path(args.receipt_file).is_absolute() else Path(args.receipt_file), {})
    return {}


def compact_provider(provider_node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": provider_node.get("node_id"),
        "node_name": provider_node.get("node_name"),
        "capabilities": [
            {"id": cap.get("id"), "score": cap.get("score")}
            for cap in provider_node.get("capabilities", [])
            if isinstance(cap, dict)
        ][:8],
    }


def receipt_hash(receipt: dict[str, Any]) -> str:
    return stable_id("work_receipt", receipt)


def stable_id(prefix: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(canonical).hexdigest()[:24]}"


def receipt_seen(path: Path, receipt_id: str) -> bool:
    if not path.exists():
        return False
    for row in read_jsonl_tail(path, 10_000):
        if row.get("receipt_id") == receipt_id:
            return True
    return False


def path_from_policy(policy: dict[str, Any], keys: list[str], default: str) -> Path:
    value = str(get_path(policy, keys, default))
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def value_int(payload: dict[str, Any], key: str, default: int) -> int:
    return clamp_int(payload.get(key, default), 0, 1_000_000_000)


def clamp_int(value: Any, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = lower
    return max(lower, min(upper, parsed))


def parse_json(raw: str, default: Any) -> Any:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, dict) else default


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
