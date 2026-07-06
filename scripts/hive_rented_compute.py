"""Dry-run-first rented compute and storage expansion for Project Theseus Hive.

This is the cloud/provider edge of the Hive. It plans when to rent external
capacity, writes auditable reports, and can launch only explicitly approved
provider commands. AWS EC2 is the first concrete adapter; other providers share
the same profile/condition surface until their launch adapters are promoted.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
POLICY_PATH = CONFIGS / "hive_policy.json"
LOCAL_CONFIG_PATH = CONFIGS / "hive_rented_compute.local.json"
DEFAULT_STATUS_PATH = REPORTS / "hive_rented_compute_status.json"
DEFAULT_PLAN_PATH = REPORTS / "hive_rented_compute_plan.json"
DEFAULT_LAUNCH_PATH = REPORTS / "hive_rented_compute_launch.json"
DEFAULT_LEDGER_PATH = REPORTS / "hive_rented_compute_ledger.jsonl"

sys.path.insert(0, str(ROOT / "scripts"))
try:
    import hive_profiles  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - status still works without profile helper
    hive_profiles = None  # type: ignore


from hive_rented_compute_profiles import (  # noqa: E402
    DEFAULT_ALLOWED_TASK_KINDS,
    canonical_provider,
    configured_profiles,
    default_local_config,
    default_plan_hours,
    default_profile_name,
    default_task_kind,
    find_profile,
    normalize_profile,
    profile_hourly_estimate,
    provider_catalog,
    provider_default_region,
    provider_required_fields,
    provider_status,
    safe_id,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan and explicitly launch rented Hive compute/storage.")
    parser.add_argument("--policy", default=str(POLICY_PATH.relative_to(ROOT)))
    parser.add_argument("--config", default=str(LOCAL_CONFIG_PATH.relative_to(ROOT)))
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status", help="Show rented-compute configuration, prerequisites, and last plan.")
    status.add_argument("--out", default=str(DEFAULT_STATUS_PATH.relative_to(ROOT)))

    init = sub.add_parser("init", help="Create an ignored local rented-compute profile template.")
    init.add_argument("--provider", default="aws_ec2")
    init.add_argument("--name", default="")
    init.add_argument("--repo-url", default="")
    init.add_argument("--branch", default="")
    init.add_argument("--region", default="us-east-1")
    init.add_argument("--out", default=str(LOCAL_CONFIG_PATH.relative_to(ROOT)))
    init.add_argument("--overwrite", action="store_true")

    plan = sub.add_parser("plan", help="Build a dry-run plan for renting compute or storage.")
    plan.add_argument("--profile", default="aws-gpu-nightly")
    plan.add_argument("--task-kind", default="cuda_training_chunk")
    plan.add_argument("--hours", type=float, default=4.0)
    plan.add_argument("--estimated-hourly-usd", type=float, default=0.0)
    plan.add_argument("--ignore-conditions", action="store_true")
    plan.add_argument("--out", default=str(DEFAULT_PLAN_PATH.relative_to(ROOT)))

    launch = sub.add_parser("launch", help="Launch an approved plan. Dry-run unless --execute is present.")
    launch.add_argument("--plan", default=str(DEFAULT_PLAN_PATH.relative_to(ROOT)))
    launch.add_argument("--execute", action="store_true")
    launch.add_argument("--out", default=str(DEFAULT_LAUNCH_PATH.relative_to(ROOT)))

    stop = sub.add_parser("stop", help="Terminate/release rented capacity. Dry-run unless --execute is present.")
    stop.add_argument("--profile", default="")
    stop.add_argument("--provider", default="aws_ec2")
    stop.add_argument("--instance-id", default="")
    stop.add_argument("--bucket", default="")
    stop.add_argument("--resource-name", default="")
    stop.add_argument("--execute", action="store_true")
    stop.add_argument("--out", default=str(DEFAULT_LAUNCH_PATH.relative_to(ROOT)))

    args = parser.parse_args()
    policy = read_json(resolve_path(args.policy), {})
    config = read_json(resolve_path(args.config), {})
    if args.command in {None, "status"}:
        report = status_report(policy=policy, config=config, write_report=True, out=args.out)
    elif args.command == "init":
        report = init_config(args, policy=policy)
    elif args.command == "plan":
        report = build_plan(
            profile_name=args.profile,
            task_kind=args.task_kind,
            hours=float(args.hours or 0),
            estimated_hourly_usd=float(args.estimated_hourly_usd or 0),
            ignore_conditions=bool(args.ignore_conditions),
            policy=policy,
            config=config,
            out=args.out,
        )
    elif args.command == "launch":
        report = launch_plan(plan_path=resolve_path(args.plan), execute=bool(args.execute), out=args.out)
    elif args.command == "stop":
        report = stop_capacity(
            profile_name=args.profile,
            provider=args.provider,
            instance_id=args.instance_id,
            bucket=args.bucket,
            resource_name=args.resource_name,
            execute=bool(args.execute),
            policy=policy,
            config=config,
            out=args.out,
        )
    else:
        parser.print_help()
        return 2
    print(json.dumps(public_report(report), indent=2))
    return 0 if report.get("ok", True) else 2


def status_report(
    *,
    policy: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    write_report: bool = True,
    out: str | Path = DEFAULT_STATUS_PATH,
) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    config = config or read_json(LOCAL_CONFIG_PATH, {})
    profiles = configured_profiles(policy, config)
    providers = provider_status()
    aws = providers.get("aws_ec2", {})
    last_plan = read_json(DEFAULT_PLAN_PATH, {})
    report = {
        "ok": True,
        "policy": "project_theseus_hive_rented_compute_status_v0",
        "created_utc": now(),
        "enabled": bool(get_path(policy, ["rented_compute", "enabled"], True)),
        "local_config_path": rel(LOCAL_CONFIG_PATH),
        "configured_profile_count": len(profiles),
        "profiles": [public_profile(profile) for profile in profiles],
        "providers": provider_catalog(providers),
        "active_hive": active_hive_public(),
        "prerequisites": {
            "aws_cli_installed": aws.get("cli_installed"),
            "aws_credentials_hint": aws.get("credentials_hint"),
            "gcloud_cli_installed": get_path(providers, ["gcp_compute", "cli_installed"], False),
            "azure_cli_installed": get_path(providers, ["azure_vm", "cli_installed"], False),
            "vastai_cli_installed": get_path(providers, ["vast_ai", "cli_installed"], False),
            "curl_installed": get_path(providers, ["runpod", "cli_installed"], False),
            "git_remote_available": bool(default_repo_url()),
            "join_config_available": bool(active_hive_config().get("join_token")),
        },
        "last_plan": compact_plan(last_plan),
        "safety": safety_summary(),
        "next_actions": rented_compute_next_actions(profiles, providers, last_plan),
        "external_inference_calls": 0,
    }
    if write_report:
        write_json(resolve_path(out), report)
    return report


def init_config(args: argparse.Namespace, *, policy: dict[str, Any]) -> dict[str, Any]:
    provider = canonical_provider(args.provider)
    name = safe_id(args.name or default_profile_name(provider))
    path = resolve_path(args.out)
    if path.exists() and not args.overwrite:
        return {
            "ok": True,
            "policy": "project_theseus_hive_rented_compute_init_v0",
            "created_utc": now(),
            "created": False,
            "path": rel(path),
            "message": "Local rented-compute config already exists. Use --overwrite to replace it.",
            "next_commands": [
                f"python scripts/hive_rented_compute.py plan --profile {name}",
                f"python scripts/theseus_cli.py rent plan --profile {name}",
            ],
        }
    payload = default_local_config(
        provider=provider,
        name=name,
        region=provider_default_region(provider, args.region),
        repo_url=args.repo_url or default_repo_url(),
        branch=args.branch or default_branch(),
    )
    write_json(path, payload)
    report = {
        "ok": True,
        "policy": "project_theseus_hive_rented_compute_init_v0",
        "created_utc": now(),
        "created": True,
        "path": rel(path),
        "profile": public_profile(payload["profiles"][0]),
        "security_note": "This file is ignored by git. Review AMI, subnet, security group, key, budget, and invite settings before launch.",
        "next_commands": [
            f"python scripts/hive_rented_compute.py plan --profile {name} --task-kind {default_task_kind(provider)} --hours {default_plan_hours(provider)}",
            f"python scripts/hive_rented_compute.py launch --plan {rel(DEFAULT_PLAN_PATH)} --execute",
        ],
    }
    write_json(REPORTS / "hive_rented_compute_init.json", report)
    return report


def build_plan(
    *,
    profile_name: str,
    task_kind: str,
    hours: float,
    estimated_hourly_usd: float,
    ignore_conditions: bool,
    policy: dict[str, Any],
    config: dict[str, Any],
    out: str | Path,
) -> dict[str, Any]:
    profile = find_profile(policy, config, profile_name)
    if not profile:
        report = {
            "ok": False,
            "policy": "project_theseus_hive_rented_compute_plan_v0",
            "created_utc": now(),
            "error": "profile_not_found",
            "profile": profile_name,
            "available_profiles": [row.get("name") for row in configured_profiles(policy, config)],
            "next_command": "python scripts/hive_rented_compute.py init --provider aws_ec2 --name aws-gpu-nightly",
        }
        write_json(resolve_path(out), report)
        return report
    profile = normalize_profile(profile)
    provider = canonical_provider(str(profile.get("provider") or ""))
    hours = max(0.25, min(float(hours or 0), float(profile.get("max_session_hours") or 12)))
    estimated_hourly_usd = estimated_hourly_usd or profile_hourly_estimate(profile)
    condition_report = evaluate_conditions(profile, task_kind, hours, estimated_hourly_usd)
    if ignore_conditions:
        condition_report["ignored"] = True
        condition_report["passed"] = True
        condition_report["blockers"] = []
    if provider == "aws_ec2":
        cloud = aws_ec2_plan(profile, task_kind=task_kind, hours=hours, hourly=estimated_hourly_usd)
    elif provider == "aws_s3":
        cloud = aws_s3_plan(profile, hours=hours, hourly=estimated_hourly_usd)
    elif provider == "gcp_compute":
        cloud = gcp_compute_plan(profile, task_kind=task_kind, hours=hours, hourly=estimated_hourly_usd)
    elif provider == "gcp_gcs":
        cloud = gcp_gcs_plan(profile, hours=hours, hourly=estimated_hourly_usd)
    elif provider == "azure_vm":
        cloud = azure_vm_plan(profile, task_kind=task_kind, hours=hours, hourly=estimated_hourly_usd)
    elif provider == "azure_blob":
        cloud = azure_blob_plan(profile, hours=hours, hourly=estimated_hourly_usd)
    elif provider == "runpod":
        cloud = runpod_plan(profile, task_kind=task_kind, hours=hours, hourly=estimated_hourly_usd)
    elif provider == "vast_ai":
        cloud = vast_ai_plan(profile, task_kind=task_kind, hours=hours, hourly=estimated_hourly_usd)
    else:
        cloud = planning_only_provider_plan(profile, provider)
    hard_blockers = list(cloud.get("blockers") or [])
    if not condition_report.get("passed"):
        hard_blockers.extend(condition_report.get("blockers") or [])
    allowed = bool(profile.get("enabled", True)) and bool(condition_report.get("passed")) and not hard_blockers
    decision = "rent" if allowed else "skip"
    report = {
        "ok": True,
        "policy": "project_theseus_hive_rented_compute_plan_v0",
        "created_utc": now(),
        "profile": public_profile(profile),
        "provider": provider,
        "task_kind": task_kind,
        "hours": hours,
        "estimated_hourly_usd": estimated_hourly_usd,
        "estimated_total_usd": round(hours * estimated_hourly_usd, 4),
        "decision": decision,
        "conditions": condition_report,
        "cloud_plan": cloud,
        "safety": safety_summary(),
        "next_actions": plan_next_actions(decision, hard_blockers, cloud),
        "external_inference_calls": 0,
    }
    write_json(resolve_path(out), report)
    append_jsonl(
        DEFAULT_LEDGER_PATH,
        ledger_event(
            "plan",
            {
                "decision": decision,
                "profile": profile.get("name"),
                "provider": provider,
                "task_kind": task_kind,
                "estimated_total_usd": report.get("estimated_total_usd"),
            },
        ),
    )
    return report


def launch_plan(*, plan_path: Path, execute: bool, out: str | Path) -> dict[str, Any]:
    plan = read_json(plan_path, {})
    cloud = plan.get("cloud_plan") if isinstance(plan.get("cloud_plan"), dict) else {}
    commands = cloud_commands(cloud, "launch")
    command = commands[0] if commands else []
    blockers = list(plan.get("conditions", {}).get("blockers") or []) + list(cloud.get("blockers") or [])
    if plan.get("decision") != "rent":
        blockers.append(f"plan_decision_is_{plan.get('decision') or 'unknown'}")
    if not command:
        blockers.append("launch_command_missing")
    report: dict[str, Any] = {
        "ok": not blockers,
        "policy": "project_theseus_hive_rented_compute_launch_v0",
        "created_utc": now(),
        "execute": bool(execute),
        "plan_path": rel(plan_path),
        "decision": plan.get("decision"),
        "provider": plan.get("provider"),
        "profile_name": get_path(plan, ["profile", "name"], ""),
        "command_preview": redact_command(command),
        "command_sequence_preview": [redact_command(row) for row in commands],
        "blockers": blockers,
        "external_inference_calls": 0,
    }
    if blockers:
        report["message"] = "Plan is not launchable. Fix blockers or rebuild with --ignore-conditions after operator review."
        write_json(resolve_path(out), report)
        append_jsonl(DEFAULT_LEDGER_PATH, ledger_event("launch_blocked", {"provider": plan.get("provider"), "blockers": blockers}))
        return report
    if not execute:
        report.update(
            {
                "ok": True,
                "dry_run": True,
                "message": "Dry run only. Add --execute to launch rented capacity.",
            }
        )
        write_json(resolve_path(out), report)
        append_jsonl(
            DEFAULT_LEDGER_PATH,
            ledger_event(
                "launch_dry_run",
                {
                    "provider": plan.get("provider"),
                    "profile": get_path(plan, ["profile", "name"], ""),
                    "estimated_total_usd": plan.get("estimated_total_usd"),
                },
            ),
        )
        return report
    result = run_provider_commands(commands, timeout=180)
    report.update(
        {
            "ok": result.get("returncode") == 0,
            "dry_run": False,
            "provider_result": result,
        }
    )
    write_json(resolve_path(out), report)
    append_jsonl(
        DEFAULT_LEDGER_PATH,
        ledger_event(
            "launch_execute",
            {
                "provider": plan.get("provider"),
                "ok": report.get("ok"),
                "returncode": result.get("returncode"),
                "estimated_total_usd": plan.get("estimated_total_usd"),
            },
        ),
    )
    return report


def stop_capacity(
    *,
    profile_name: str,
    provider: str,
    instance_id: str,
    bucket: str,
    execute: bool,
    policy: dict[str, Any],
    config: dict[str, Any],
    out: str | Path,
    resource_name: str = "",
) -> dict[str, Any]:
    profile = find_profile(policy, config, profile_name) if profile_name else {}
    provider = canonical_provider(str((profile or {}).get("provider") or provider))
    if provider == "aws_ec2":
        instance_id = instance_id or resource_name
        if not instance_id:
            report = {"ok": False, "error": "instance_id_required", "policy": "project_theseus_hive_rented_compute_stop_v0", "created_utc": now()}
            write_json(resolve_path(out), report)
            return report
        command = aws_prefix(profile or {}) + ["ec2", "terminate-instances", "--instance-ids", instance_id]
        region = str((profile or {}).get("region") or "")
        if region:
            command += ["--region", region]
    elif provider == "aws_s3":
        bucket = bucket or resource_name
        if not bucket:
            report = {"ok": False, "error": "bucket_required", "policy": "project_theseus_hive_rented_compute_stop_v0", "created_utc": now()}
            write_json(resolve_path(out), report)
            return report
        command = aws_prefix(profile or {}) + ["s3", "rb", f"s3://{bucket}"]
    elif provider == "gcp_compute":
        name = resource_name or str((profile or {}).get("instance_name") or (profile or {}).get("name") or "")
        zone = str((profile or {}).get("zone") or "")
        project = str((profile or {}).get("project") or "")
        if not name or not zone:
            report = {"ok": False, "error": "instance_name_and_zone_required", "policy": "project_theseus_hive_rented_compute_stop_v0", "created_utc": now()}
            write_json(resolve_path(out), report)
            return report
        command = ["gcloud", "compute", "instances", "delete", name, "--zone", zone, "--quiet"]
        if project:
            command += ["--project", project]
    elif provider == "gcp_gcs":
        bucket = bucket or resource_name or str((profile or {}).get("bucket") or "")
        if not bucket:
            report = {"ok": False, "error": "bucket_required", "policy": "project_theseus_hive_rented_compute_stop_v0", "created_utc": now()}
            write_json(resolve_path(out), report)
            return report
        command = ["gcloud", "storage", "rm", "--recursive", f"gs://{bucket}"]
    elif provider == "azure_vm":
        name = resource_name or str((profile or {}).get("vm_name") or (profile or {}).get("name") or "")
        group = str((profile or {}).get("resource_group") or "")
        if not name or not group:
            report = {"ok": False, "error": "vm_name_and_resource_group_required", "policy": "project_theseus_hive_rented_compute_stop_v0", "created_utc": now()}
            write_json(resolve_path(out), report)
            return report
        command = ["az", "vm", "delete", "--resource-group", group, "--name", name, "--yes"]
    elif provider == "azure_blob":
        container = resource_name or str((profile or {}).get("container") or "")
        account = str((profile or {}).get("storage_account") or "")
        if not container or not account:
            report = {"ok": False, "error": "storage_account_and_container_required", "policy": "project_theseus_hive_rented_compute_stop_v0", "created_utc": now()}
            write_json(resolve_path(out), report)
            return report
        command = ["az", "storage", "container", "delete", "--account-name", account, "--name", container]
    elif provider == "runpod":
        pod_id = resource_name or instance_id or str((profile or {}).get("pod_id") or "")
        if not pod_id:
            report = {"ok": False, "error": "pod_id_required", "policy": "project_theseus_hive_rented_compute_stop_v0", "created_utc": now()}
            write_json(resolve_path(out), report)
            return report
        command = ["curl", "--request", "DELETE", "--url", f"https://rest.runpod.io/v1/pods/{pod_id}", "--header", "Authorization: Bearer {env:RUNPOD_API_KEY}"]
    elif provider == "vast_ai":
        vast_id = resource_name or instance_id or str((profile or {}).get("instance_id") or "")
        if not vast_id:
            report = {"ok": False, "error": "vast_instance_id_required", "policy": "project_theseus_hive_rented_compute_stop_v0", "created_utc": now()}
            write_json(resolve_path(out), report)
            return report
        command = ["vastai", "destroy", "instance", vast_id, "--raw"]
    else:
        command = []
    if not command:
        report = {
            "ok": False,
            "policy": "project_theseus_hive_rented_compute_stop_v0",
            "created_utc": now(),
            "error": "provider_stop_adapter_not_implemented",
            "provider": provider,
        }
        write_json(resolve_path(out), report)
        return report
    report: dict[str, Any] = {
        "ok": True,
        "policy": "project_theseus_hive_rented_compute_stop_v0",
        "created_utc": now(),
        "execute": bool(execute),
        "provider": provider,
        "command_preview": redact_command(command),
    }
    if not execute:
        report["dry_run"] = True
        report["message"] = "Dry run only. Add --execute to release rented capacity."
        write_json(resolve_path(out), report)
        append_jsonl(DEFAULT_LEDGER_PATH, ledger_event("stop_dry_run", {"provider": provider, "instance_id": instance_id, "bucket": bucket}))
        return report
    result = run_provider_commands([command], timeout=120)
    report.update({"dry_run": False, "ok": result.get("returncode") == 0, "provider_result": result})
    write_json(resolve_path(out), report)
    append_jsonl(DEFAULT_LEDGER_PATH, ledger_event("stop_execute", {"provider": provider, "ok": report.get("ok"), "returncode": result.get("returncode")}))
    return report


def aws_ec2_plan(profile: dict[str, Any], *, task_kind: str, hours: float, hourly: float) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    ami = str(profile.get("ami_id") or "")
    instance_type = str(profile.get("instance_type") or "")
    region = str(profile.get("region") or "")
    if not shutil.which("aws"):
        warnings.append("aws_cli_not_installed_on_planning_node")
    if not ami or "REPLACE" in ami.upper():
        blockers.append("ami_id_required")
    if not instance_type:
        blockers.append("instance_type_required")
    if not region:
        blockers.append("region_required")
    if not (profile.get("subnet_id") or profile.get("allow_default_vpc")):
        warnings.append("subnet_id_not_set_aws_may_use_default_vpc_if_available")
    security_group_ids = profile.get("security_group_ids") if isinstance(profile.get("security_group_ids"), list) else []
    if not security_group_ids and not profile.get("allow_default_security_group"):
        warnings.append("security_group_ids_empty_aws_may_use_default_security_group")
    user_data = write_user_data(profile, task_kind=task_kind)
    tags = profile_tags(profile, task_kind=task_kind, hours=hours, hourly=hourly)
    command = aws_prefix(profile) + [
        "ec2",
        "run-instances",
        "--image-id",
        ami or "ami-REPLACE_ME",
        "--instance-type",
        instance_type or "g5.xlarge",
        "--count",
        "1",
        "--region",
        region or "us-east-1",
        "--user-data",
        f"file://{user_data['path']}",
        "--block-device-mappings",
        json.dumps(block_device_mapping(profile), separators=(",", ":")),
        "--tag-specifications",
        tag_specification("instance", tags),
        tag_specification("volume", tags),
    ]
    if profile.get("key_name"):
        command += ["--key-name", str(profile.get("key_name"))]
    if profile.get("subnet_id"):
        command += ["--subnet-id", str(profile.get("subnet_id"))]
    if security_group_ids:
        command += ["--security-group-ids", *[str(item) for item in security_group_ids]]
    if profile.get("iam_instance_profile"):
        command += ["--iam-instance-profile", f"Name={profile.get('iam_instance_profile')}"]
    if bool(profile.get("spot", True)):
        command += ["--instance-market-options", json.dumps(spot_options(profile), separators=(",", ":"))]
    return {
        "adapter": "aws_ec2",
        "adapter_maturity": "R2_sandboxed_cloud_launch_adapter",
        "blockers": blockers,
        "warnings": warnings,
        "launch_command": command,
        "user_data_path": user_data["path"],
        "user_data_contains_secret": user_data["contains_secret"],
        "user_data_redacted_preview": user_data["redacted_preview"],
        "expected_node_join": "private_hive" if user_data["contains_invite"] else "manual_or_relay_config_required",
        "stop_command_template": redact_command(aws_prefix(profile) + ["ec2", "terminate-instances", "--instance-ids", "INSTANCE_ID", "--region", region or "us-east-1"]),
        "provider_docs": {
            "run_instances": "https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html",
            "user_data": "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html",
            "spot": "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-spot-instances-work.html",
        },
    }


def aws_s3_plan(profile: dict[str, Any], *, hours: float, hourly: float) -> dict[str, Any]:
    bucket = str(profile.get("bucket") or "")
    region = str(profile.get("region") or "us-east-1")
    blockers = [] if bucket and not is_placeholder(bucket) else ["bucket_required"]
    command = aws_prefix(profile) + ["s3api", "create-bucket", "--bucket", bucket or "theseus-hive-REPLACE-ME", "--region", region]
    if region != "us-east-1":
        command += ["--create-bucket-configuration", json.dumps({"LocationConstraint": region})]
    return {
        "adapter": "aws_s3",
        "adapter_maturity": "R1_planning_adapter_with_explicit_execute",
        "blockers": blockers,
        "warnings": [
            "storage_adapter_creates_bucket_only; artifact sync policy still controls what may be uploaded",
            "configure lifecycle/cost controls in AWS before large checkpoint or dataset storage",
        ],
        "launch_command": command,
        "stop_command_template": redact_command(aws_prefix(profile) + ["s3", "rb", f"s3://{bucket or 'BUCKET'}"]),
        "estimated_total_usd": round(hours * hourly, 4),
    }


def gcp_compute_plan(profile: dict[str, Any], *, task_kind: str, hours: float, hourly: float) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    name = str(profile.get("instance_name") or profile.get("name") or "theseus-gcp-worker")
    project = str(profile.get("project") or "")
    zone = str(profile.get("zone") or "")
    machine_type = str(profile.get("machine_type") or "")
    image_family = str(profile.get("image_family") or "ubuntu-2204-lts")
    image_project = str(profile.get("image_project") or "ubuntu-os-cloud")
    accelerator = profile.get("accelerator") if isinstance(profile.get("accelerator"), dict) else {}
    if not shutil.which("gcloud"):
        warnings.append("gcloud_cli_not_installed_on_planning_node")
    if not project or is_placeholder(project):
        blockers.append("gcp_project_required")
    if not zone:
        blockers.append("gcp_zone_required")
    if not machine_type:
        blockers.append("gcp_machine_type_required")
    user_data = write_user_data(profile, task_kind=task_kind)
    command = [
        "gcloud",
        "compute",
        "instances",
        "create",
        name,
        "--project",
        project or "PROJECT",
        "--zone",
        zone or "ZONE",
        "--machine-type",
        machine_type or "g2-standard-4",
        "--boot-disk-size",
        f"{int(profile.get('root_volume_gib') or 200)}GB",
        "--image-family",
        image_family,
        "--image-project",
        image_project,
        "--metadata-from-file",
        f"startup-script={user_data['path']}",
        "--labels",
        comma_labels(profile_tags(profile, task_kind=task_kind, hours=hours, hourly=hourly)),
    ]
    if profile.get("network"):
        command += ["--network", str(profile.get("network"))]
    if profile.get("subnet"):
        command += ["--subnet", str(profile.get("subnet"))]
    if profile.get("no_external_ip", True):
        command += ["--no-address"]
    if accelerator.get("type") and accelerator.get("count"):
        command += ["--accelerator", f"type={accelerator.get('type')},count={accelerator.get('count')}"]
        command += ["--maintenance-policy", "TERMINATE"]
    if bool(profile.get("spot", True)):
        command += ["--provisioning-model", "SPOT", "--instance-termination-action", "DELETE"]
    return {
        "adapter": "gcp_compute",
        "adapter_maturity": "R2_sandboxed_cloud_launch_adapter",
        "blockers": blockers,
        "warnings": warnings,
        "launch_command": command,
        "user_data_path": user_data["path"],
        "user_data_contains_secret": user_data["contains_secret"],
        "user_data_redacted_preview": user_data["redacted_preview"],
        "stop_command_template": redact_command(["gcloud", "compute", "instances", "delete", name, "--zone", zone or "ZONE", "--project", project or "PROJECT", "--quiet"]),
        "provider_docs": {
            "create_instance": "https://cloud.google.com/sdk/gcloud/reference/compute/instances/create",
            "startup_scripts": "https://cloud.google.com/compute/docs/instances/startup-scripts/linux",
        },
    }


def gcp_gcs_plan(profile: dict[str, Any], *, hours: float, hourly: float) -> dict[str, Any]:
    bucket = str(profile.get("bucket") or "")
    location = str(profile.get("location") or profile.get("region") or "US")
    project = str(profile.get("project") or "")
    blockers = [] if bucket and not is_placeholder(bucket) else ["bucket_required"]
    if not project or is_placeholder(project):
        blockers.append("gcp_project_required")
    command = ["gcloud", "storage", "buckets", "create", f"gs://{bucket or 'theseus-hive-REPLACE-ME'}", "--location", location]
    if project:
        command += ["--project", project]
    return {
        "adapter": "gcp_gcs",
        "adapter_maturity": "R1_planning_adapter_with_explicit_execute",
        "blockers": blockers,
        "warnings": ["storage_adapter_creates_bucket_only; Hive artifact sync still controls uploads"],
        "launch_command": command,
        "stop_command_template": redact_command(["gcloud", "storage", "rm", "--recursive", f"gs://{bucket or 'BUCKET'}"]),
        "estimated_total_usd": round(hours * hourly, 4),
    }


def azure_vm_plan(profile: dict[str, Any], *, task_kind: str, hours: float, hourly: float) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    group = str(profile.get("resource_group") or "")
    name = str(profile.get("vm_name") or profile.get("name") or "theseus-azure-worker")
    location = str(profile.get("location") or profile.get("region") or "")
    size = str(profile.get("vm_size") or profile.get("instance_type") or "")
    image = str(profile.get("image") or "Ubuntu2204")
    admin = str(profile.get("admin_username") or "theseus")
    if not shutil.which("az"):
        warnings.append("azure_cli_not_installed_on_planning_node")
    if not group or is_placeholder(group):
        blockers.append("azure_resource_group_required")
    if not location:
        blockers.append("azure_location_required")
    if not size:
        blockers.append("azure_vm_size_required")
    user_data = write_user_data(profile, task_kind=task_kind)
    command = [
        "az",
        "vm",
        "create",
        "--resource-group",
        group or "RESOURCE_GROUP",
        "--name",
        name,
        "--image",
        image,
        "--size",
        size or "Standard_NC4as_T4_v3",
        "--admin-username",
        admin,
        "--custom-data",
        user_data["path"],
        "--os-disk-size-gb",
        str(int(profile.get("root_volume_gib") or 200)),
        "--tags",
        *space_tags(profile_tags(profile, task_kind=task_kind, hours=hours, hourly=hourly)),
    ]
    if location:
        command += ["--location", location]
    if profile.get("ssh_key_value"):
        command += ["--ssh-key-values", str(profile.get("ssh_key_value"))]
    else:
        command += ["--generate-ssh-keys"]
    if profile.get("vnet_name"):
        command += ["--vnet-name", str(profile.get("vnet_name"))]
    if profile.get("subnet"):
        command += ["--subnet", str(profile.get("subnet"))]
    if bool(profile.get("spot", True)):
        command += ["--priority", "Spot", "--eviction-policy", str(profile.get("eviction_policy") or "Deallocate")]
        max_price = profile.get("max_price_usd_per_hour") or profile.get("max_price")
        if max_price:
            command += ["--max-price", str(max_price)]
    return {
        "adapter": "azure_vm",
        "adapter_maturity": "R2_sandboxed_cloud_launch_adapter",
        "blockers": blockers,
        "warnings": warnings,
        "launch_command": command,
        "user_data_path": user_data["path"],
        "user_data_contains_secret": user_data["contains_secret"],
        "user_data_redacted_preview": user_data["redacted_preview"],
        "stop_command_template": redact_command(["az", "vm", "delete", "--resource-group", group or "RESOURCE_GROUP", "--name", name, "--yes"]),
        "provider_docs": {
            "az_vm": "https://learn.microsoft.com/cli/azure/vm",
            "cloud_init": "https://learn.microsoft.com/azure/virtual-machines/linux/using-cloud-init",
        },
    }


def azure_blob_plan(profile: dict[str, Any], *, hours: float, hourly: float) -> dict[str, Any]:
    blockers: list[str] = []
    account = str(profile.get("storage_account") or "")
    container = str(profile.get("container") or "")
    group = str(profile.get("resource_group") or "")
    location = str(profile.get("location") or profile.get("region") or "")
    if not account or is_placeholder(account):
        blockers.append("azure_storage_account_required")
    if not container or is_placeholder(container):
        blockers.append("azure_container_required")
    if not group or is_placeholder(group):
        blockers.append("azure_resource_group_required")
    if not location:
        blockers.append("azure_location_required")
    commands = [
        ["az", "storage", "account", "create", "--name", account or "STORAGE_ACCOUNT", "--resource-group", group or "RESOURCE_GROUP", "--location", location or "LOCATION", "--sku", str(profile.get("sku") or "Standard_LRS")],
        ["az", "storage", "container", "create", "--account-name", account or "STORAGE_ACCOUNT", "--name", container or "theseus-hive"],
    ]
    return {
        "adapter": "azure_blob",
        "adapter_maturity": "R1_planning_adapter_with_explicit_execute",
        "blockers": blockers,
        "warnings": ["storage_adapter_creates_account_container_only; Hive artifact sync still controls uploads"],
        "launch_commands": commands,
        "launch_command": commands[0],
        "stop_command_template": redact_command(["az", "storage", "container", "delete", "--account-name", account or "STORAGE_ACCOUNT", "--name", container or "CONTAINER"]),
        "estimated_total_usd": round(hours * hourly, 4),
    }


def runpod_plan(profile: dict[str, Any], *, task_kind: str, hours: float, hourly: float) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if not shutil.which("curl"):
        warnings.append("curl_not_installed_on_planning_node")
    if not os.environ.get("RUNPOD_API_KEY"):
        warnings.append("RUNPOD_API_KEY_env_not_set_on_planning_node")
    gpu_type = str(profile.get("gpu_type") or "")
    if not gpu_type:
        blockers.append("runpod_gpu_type_required")
    payload_path = write_runpod_payload(profile, task_kind=task_kind, hours=hours)
    command = [
        "curl",
        "--request",
        "POST",
        "--url",
        "https://rest.runpod.io/v1/pods",
        "--header",
        "Authorization: Bearer {env:RUNPOD_API_KEY}",
        "--header",
        "Content-Type: application/json",
        "--data",
        f"@{payload_path}",
    ]
    return {
        "adapter": "runpod",
        "adapter_maturity": "R1_api_launch_adapter_requires_profile_review",
        "blockers": blockers,
        "warnings": warnings,
        "launch_command": command,
        "payload_path": payload_path,
        "payload_contains_secret": True,
        "stop_command_template": redact_command(["curl", "--request", "DELETE", "--url", "https://rest.runpod.io/v1/pods/POD_ID", "--header", "Authorization: Bearer {env:RUNPOD_API_KEY}"]),
        "provider_docs": {"create_pod": "https://docs.runpod.io/api-reference/pods/POST/pods"},
        "estimated_total_usd": round(hours * hourly, 4),
    }


def vast_ai_plan(profile: dict[str, Any], *, task_kind: str, hours: float, hourly: float) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    offer_id = str(profile.get("offer_id") or "")
    image = str(profile.get("image") or "pytorch/pytorch:latest")
    if not shutil.which("vastai"):
        warnings.append("vastai_cli_not_installed_on_planning_node")
    if not offer_id or is_placeholder(offer_id):
        blockers.append("vast_offer_id_required")
    bootstrap = bootstrap_one_liner(profile, task_kind=task_kind)
    command = [
        "vastai",
        "create",
        "instance",
        offer_id or "OFFER_ID",
        "--image",
        image,
        "--disk",
        str(int(profile.get("disk_gib") or profile.get("root_volume_gib") or 120)),
        "--onstart-cmd",
        "bash",
        "--raw",
    ]
    if bool(profile.get("spot", True)):
        bid = profile.get("bid_price") or profile.get("max_price_usd_per_hour")
        if bid:
            command += ["--bid_price", str(bid)]
    command += ["--args", "-lc", bootstrap]
    return {
        "adapter": "vast_ai",
        "adapter_maturity": "R1_cli_launch_adapter_requires_offer_review",
        "blockers": blockers,
        "warnings": warnings,
        "launch_command": command,
        "bootstrap_contains_secret": "{env:" not in bootstrap and bool(active_hive_config().get("join_token")),
        "stop_command_template": redact_command(["vastai", "destroy", "instance", "INSTANCE_ID", "--raw"]),
        "provider_docs": {"create_instance": "https://docs.vast.ai/cli/reference/create-instance"},
        "estimated_total_usd": round(hours * hourly, 4),
    }


def planning_only_provider_plan(profile: dict[str, Any], provider: str) -> dict[str, Any]:
    return {
        "adapter": provider,
        "adapter_maturity": "R0_contract_only",
        "blockers": ["provider_launch_adapter_not_implemented"],
        "warnings": [
            "Profile is preserved so the same conditions/budget surface can be used when the provider adapter is implemented.",
        ],
        "launch_command": [],
        "required_profile_fields": provider_required_fields(provider),
    }


def write_user_data(profile: dict[str, Any], *, task_kind: str) -> dict[str, Any]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    profile_name = safe_id(str(profile.get("name") or "rented-worker"))
    path = REPORTS / f"hive_rented_compute_user_data_{profile_name}.sh"
    bootstrap = profile.get("bootstrap") if isinstance(profile.get("bootstrap"), dict) else {}
    repo_url = str(bootstrap.get("repo_url") or default_repo_url() or "REPLACE_WITH_PRIVATE_REPO_URL")
    branch = str(bootstrap.get("branch") or default_branch() or "main")
    install_root = str(bootstrap.get("install_root") or "/opt/project-theseus")
    invite = active_hive_config() if bool(bootstrap.get("join_from_active_hive", True)) else {}
    include_invite = bool(invite.get("join_token")) and bool(bootstrap.get("include_join_token_in_user_data", True))
    invite_b64 = base64.b64encode(json.dumps(invite, indent=2).encode("utf-8")).decode("ascii") if include_invite else ""
    relay_url = str(bootstrap.get("relay_url") or invite.get("relay_url") or "")
    coordinator_url = str(bootstrap.get("coordinator_url") or invite.get("coordinator_url") or "")
    public_worker_name = safe_id(str(profile.get("name") or socket.gethostname()))
    install_parent = install_root.rstrip("/").rsplit("/", 1)[0] or "/"
    lines = [
        "#!/usr/bin/env bash",
        "set -euxo pipefail",
        "export DEBIAN_FRONTEND=noninteractive",
        "apt-get update",
        "apt-get install -y git curl python3 python3-venv python3-pip build-essential",
        f"mkdir -p {sh_quote(install_parent)}",
        f"if [ ! -d {sh_quote(install_root + '/.git')} ]; then git clone --branch {sh_quote(branch)} {sh_quote(repo_url)} {sh_quote(install_root)}; else cd {sh_quote(install_root)} && git fetch --all && git checkout {sh_quote(branch)} && git pull --ff-only; fi",
        f"cd {sh_quote(install_root)}",
        "mkdir -p configs reports",
    ]
    if include_invite:
        lines.extend(
            [
                f"printf '%s' {sh_quote(invite_b64)} | base64 -d > /tmp/theseus_hive_invite.json",
                "chmod 600 /tmp/theseus_hive_invite.json",
            ]
        )
    install_cmd = ["./scripts/install_theseus_hive_linux.sh", "--enable-service", "--auto-update-soft"]
    if include_invite:
        install_cmd += ["--invite", "/tmp/theseus_hive_invite.json"]
    if relay_url:
        install_cmd += ["--relay-url", relay_url]
    if coordinator_url:
        install_cmd += ["--coordinator-url", coordinator_url]
    if bootstrap.get("public_contribution_mode") in {"idle", "always"}:
        install_cmd += ["--public-mode", str(bootstrap.get("public_contribution_mode")), "--allow-public", "--public-worker-name", public_worker_name]
    lines.append(" ".join(sh_quote(part) for part in install_cmd))
    lines.extend(
        [
            "python3 scripts/hive_node.py probe --out reports/hive_status.json || true",
            "python3 scripts/hive_scheduler.py --out reports/hive_scheduler.json || true",
            f"printf '%s\\n' {sh_quote('Project Theseus rented worker bootstrap finished for ' + task_kind)} > reports/hive_rented_worker_bootstrap.txt",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    redacted_lines = []
    for line in lines:
        if invite_b64 and invite_b64 in line:
            redacted_lines.append(line.replace(invite_b64, "REDACTED_INVITE_B64"))
        else:
            redacted_lines.append(line)
    return {
        "path": rel(path),
        "contains_secret": include_invite,
        "contains_invite": include_invite,
        "redacted_preview": "\n".join(redacted_lines[:18]),
    }


def bootstrap_one_liner(profile: dict[str, Any], *, task_kind: str) -> str:
    script = write_user_data(profile, task_kind=task_kind)
    script_path = resolve_path(script["path"])
    lines = script_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return " && ".join(line for line in lines if line and not line.startswith("#"))


def write_runpod_payload(profile: dict[str, Any], *, task_kind: str, hours: float) -> str:
    REPORTS.mkdir(parents=True, exist_ok=True)
    profile_name = safe_id(str(profile.get("name") or "runpod-worker"))
    path = REPORTS / f"hive_rented_compute_runpod_payload_{profile_name}.json"
    payload = {
        "name": str(profile.get("pod_name") or profile.get("name") or "theseus-runpod-worker"),
        "cloudType": str(profile.get("cloud_type") or "SECURE"),
        "computeType": "GPU",
        "gpuCount": int(profile.get("gpu_count") or 1),
        "gpuTypeIds": [str(profile.get("gpu_type") or "NVIDIA GeForce RTX 4090")],
        "gpuTypePriority": str(profile.get("gpu_type_priority") or "availability"),
        "containerDiskInGb": int(profile.get("container_disk_gib") or profile.get("root_volume_gib") or 100),
        "imageName": str(profile.get("image") or "pytorch/pytorch:latest"),
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [bootstrap_one_liner(profile, task_kind=task_kind)],
        "env": {
            "THESEUS_RENTED_WORKER": "1",
            "THESEUS_TASK_KIND": task_kind,
            "THESEUS_MAX_HOURS": str(hours),
        },
        "globalNetworking": bool(profile.get("global_networking", False)),
    }
    if profile.get("network_volume_id"):
        payload["networkVolumeId"] = str(profile.get("network_volume_id"))
    if isinstance(profile.get("runpod_payload_overrides"), dict):
        payload.update(profile["runpod_payload_overrides"])
    path.write_text(json.dumps(redact_value(payload), indent=2) + "\n", encoding="utf-8")
    private_path = REPORTS / f"hive_rented_compute_runpod_payload_{profile_name}.local.json"
    private_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return rel(private_path)


def comma_labels(tags: dict[str, str]) -> str:
    return ",".join(f"{safe_label_key(key)}={safe_label_value(value)}" for key, value in tags.items())


def space_tags(tags: dict[str, str]) -> list[str]:
    return [f"{safe_label_key(key)}={safe_label_value(value)}" for key, value in tags.items()]


def effective_conditions(profile: dict[str, Any]) -> dict[str, Any]:
    policy = read_json(POLICY_PATH, {})
    merged: dict[str, Any] = {}
    default_conditions = get_path(policy, ["rented_compute", "default_conditions"], {})
    if isinstance(default_conditions, dict):
        merged.update(default_conditions)
    policy_id = str(profile.get("policy_id") or profile.get("user_policy") or "")
    user_policies = get_path(policy, ["rented_compute", "user_policies"], {})
    if policy_id and isinstance(user_policies, dict) and isinstance(user_policies.get(policy_id), dict):
        merged.update(user_policies[policy_id])
    local_conditions = profile.get("conditions") if isinstance(profile.get("conditions"), dict) else {}
    merged.update(local_conditions)
    return merged


def evaluate_conditions(profile: dict[str, Any], task_kind: str, hours: float, hourly: float) -> dict[str, Any]:
    conditions = effective_conditions(profile)
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    allowed_kinds = profile.get("allowed_task_kinds") if isinstance(profile.get("allowed_task_kinds"), list) else DEFAULT_ALLOWED_TASK_KINDS
    allowed_task = task_kind in allowed_kinds
    add_check(checks, "allowed_task_kind", allowed_task, {"task_kind": task_kind, "allowed_task_kinds": allowed_kinds})
    if not allowed_task:
        blockers.append("task_kind_not_allowed_for_profile")
    max_hours = float(profile.get("max_session_hours") or conditions.get("max_session_hours") or 12)
    hours_ok = hours <= max_hours
    add_check(checks, "max_session_hours", hours_ok, {"hours": hours, "max_session_hours": max_hours})
    if not hours_ok:
        blockers.append("requested_hours_exceed_profile_limit")
    max_hourly = float(conditions.get("max_estimated_hourly_usd") or profile.get("max_price_usd_per_hour") or profile.get("max_estimated_hourly_usd") or 0)
    if max_hourly > 0:
        hourly_ok = hourly <= max_hourly
        add_check(checks, "max_estimated_hourly_usd", hourly_ok, {"estimated_hourly_usd": hourly, "max_estimated_hourly_usd": max_hourly})
        if not hourly_ok:
            blockers.append("estimated_hourly_cost_exceeds_limit")
    total_limit = float(conditions.get("max_total_usd_per_session") or profile.get("max_total_usd_per_session") or 0)
    if total_limit > 0:
        total_ok = hourly * hours <= total_limit
        add_check(checks, "max_total_usd_per_session", total_ok, {"estimated_total_usd": round(hourly * hours, 4), "max_total_usd_per_session": total_limit})
        if not total_ok:
            blockers.append("estimated_total_cost_exceeds_limit")
    daily_limit = float(conditions.get("max_daily_planned_usd") or 0)
    if daily_limit:
        spend = planned_spend_window(hours=24)
        daily_ok = float(spend.get("estimated_total_usd") or 0) + (hourly * hours) <= daily_limit
        add_check(checks, "max_daily_planned_usd", daily_ok, {**spend, "this_plan_usd": round(hourly * hours, 4), "max_daily_planned_usd": daily_limit})
        if not daily_ok:
            blockers.append("daily_rented_compute_budget_exceeded")
    weekly_limit = float(conditions.get("max_weekly_planned_usd") or 0)
    if weekly_limit:
        spend = planned_spend_window(hours=24 * 7)
        weekly_ok = float(spend.get("estimated_total_usd") or 0) + (hourly * hours) <= weekly_limit
        add_check(checks, "max_weekly_planned_usd", weekly_ok, {**spend, "this_plan_usd": round(hourly * hours, 4), "max_weekly_planned_usd": weekly_limit})
        if not weekly_ok:
            blockers.append("weekly_rented_compute_budget_exceeded")
    cooldown_minutes = float(conditions.get("cooldown_minutes") or 0)
    if cooldown_minutes:
        cooldown = launch_cooldown(cooldown_minutes)
        add_check(checks, "cooldown_minutes", bool(cooldown.get("passed")), cooldown)
        if not cooldown.get("passed"):
            blockers.append("rented_compute_cooldown_active")
    windows = conditions.get("time_windows") if isinstance(conditions.get("time_windows"), list) else []
    if windows:
        window_ok = any(time_window_matches(window, datetime.now().astimezone()) for window in windows if isinstance(window, dict))
        add_check(checks, "time_window", window_ok, {"time_windows": windows, "local_now": datetime.now().astimezone().isoformat(timespec="seconds")})
        if not window_ok:
            blockers.append("outside_allowed_time_window")
    if conditions.get("spot_only") is True and not bool(profile.get("spot", False)):
        add_check(checks, "spot_only", False, {"spot": bool(profile.get("spot", False))})
        blockers.append("profile_is_not_spot")
    elif conditions.get("spot_only") is True:
        add_check(checks, "spot_only", True, {"spot": True})
    if conditions.get("require_queue_pressure"):
        pressure = queue_pressure(task_kind)
        add_check(checks, "queue_pressure", bool(pressure.get("present")), pressure)
        if not pressure.get("present"):
            blockers.append("queue_pressure_missing")
    if conditions.get("require_broad_transfer_below_floor"):
        transfer = broad_transfer_pressure()
        gap = float(transfer.get("floor") or 0) - float(transfer.get("score") or 0)
        min_gap = float(conditions.get("min_transfer_floor_gap") or 0)
        transfer_ok = bool(transfer.get("below_floor")) and gap >= min_gap
        add_check(checks, "broad_transfer_below_floor", transfer_ok, {**transfer, "floor_gap": round(gap, 6), "min_transfer_floor_gap": min_gap})
        if not transfer_ok:
            blockers.append("broad_transfer_not_below_floor")
    services = conditions.get("require_services") if isinstance(conditions.get("require_services"), list) else []
    if services:
        service = service_status()
        missing = [item for item in services if not service.get(str(item))]
        add_check(checks, "require_services", not missing, {"required": services, "missing": missing, "status": service})
        if missing:
            blockers.append("required_local_service_down")
    if conditions.get("require_local_gpu_busy"):
        busy = local_gpu_busy()
        add_check(checks, "local_gpu_busy", bool(busy.get("busy")), busy)
        if not busy.get("busy"):
            blockers.append("local_gpu_not_busy")
    min_disk = float(conditions.get("min_local_disk_free_gib") or 0)
    if min_disk:
        disk = local_disk_free()
        disk_ok = float(disk.get("free_gib") or 0) >= min_disk
        add_check(checks, "min_local_disk_free_gib", disk_ok, {**disk, "min_local_disk_free_gib": min_disk})
        if not disk_ok:
            blockers.append("local_disk_below_floor")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "checks": checks,
        "signals": {
            "queue_pressure": queue_pressure(task_kind),
            "broad_transfer": broad_transfer_pressure(),
            "local_gpu": local_gpu_busy(),
        },
    }


def aws_prefix(profile: dict[str, Any]) -> list[str]:
    command = ["aws"]
    aws_profile = str(profile.get("aws_profile") or "")
    if aws_profile:
        command += ["--profile", aws_profile]
    return command


def spot_options(profile: dict[str, Any]) -> dict[str, Any]:
    opts: dict[str, Any] = {"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}}
    max_price = profile.get("max_price_usd_per_hour") or profile.get("max_price")
    if max_price:
        opts["SpotOptions"]["MaxPrice"] = str(max_price)
    return opts


def block_device_mapping(profile: dict[str, Any]) -> list[dict[str, Any]]:
    size = int(profile.get("root_volume_gib") or 200)
    device = str(profile.get("root_device_name") or "/dev/sda1")
    return [{"DeviceName": device, "Ebs": {"VolumeSize": size, "VolumeType": "gp3", "DeleteOnTermination": True}}]


def profile_tags(profile: dict[str, Any], *, task_kind: str, hours: float, hourly: float) -> dict[str, str]:
    tags: dict[str, str] = {}
    if isinstance(profile.get("tags"), dict):
        tags.update({safe_tag_key(k): safe_tag_value(v) for k, v in profile["tags"].items()})
    tags.setdefault("Project", "ProjectTheseus")
    tags.setdefault("ManagedBy", "theseus-rent")
    tags["Name"] = safe_tag_value(profile.get("name") or "theseus-rented-worker")
    tags["TheseusTaskKind"] = safe_tag_value(task_kind)
    tags["TheseusMaxHours"] = safe_tag_value(str(hours))
    tags["TheseusHourlyLimitUsd"] = safe_tag_value(str(hourly))
    active = active_hive_config()
    if active.get("hive_id"):
        tags["TheseusHiveId"] = safe_tag_value(str(active.get("hive_id")))
    return tags


def tag_specification(resource_type: str, tags: dict[str, str]) -> str:
    pairs = ",".join([f"{{Key={key},Value={value}}}" for key, value in tags.items()])
    return f"ResourceType={resource_type},Tags=[{pairs}]"


def queue_pressure(task_kind: str) -> dict[str, Any]:
    scheduler = read_json(REPORTS / "hive_scheduler.json", {})
    placements = scheduler.get("placements") if isinstance(scheduler.get("placements"), list) else []
    matching = [row for row in placements if isinstance(row, dict) and row.get("task_kind") == task_kind]
    queue_path = ROOT / str(get_path(read_json(POLICY_PATH, {}), ["node", "task_queue_path"], "reports/hive_task_queue.jsonl"))
    queued = read_jsonl_tail(queue_path, 20)
    return {
        "present": bool(matching or queued),
        "matching_scheduler_placements": len(matching),
        "queued_task_count_tail": len(queued),
        "task_kind": task_kind,
    }


def broad_transfer_pressure() -> dict[str, Any]:
    candidates = [
        read_json(REPORTS / "broad_transfer_matrix.json", {}),
        read_json(REPORTS / "student_learning_closure.json", {}),
        read_json(REPORTS / "learning_scoreboard.json", {}),
    ]
    score = None
    floor = 0.70
    source = ""
    for idx, report in enumerate(candidates):
        found_score = first_number(report, ["aggregate_pass_rate", "broad_transfer_pass_rate", "pass_rate", "score"])
        found_floor = first_number(report, ["floor", "promotion_floor", "required_floor"])
        if found_score is not None:
            score = found_score
            if found_floor is not None:
                floor = found_floor
            source = ["broad_transfer_matrix", "student_learning_closure", "learning_scoreboard"][idx]
            break
    return {
        "below_floor": score is not None and score < floor,
        "score": score,
        "floor": floor,
        "source": source,
    }


def local_gpu_busy() -> dict[str, Any]:
    status = read_json(REPORTS / "hive_status.json", {})
    gpus = get_path(status, ["resources", "nvidia", "gpus"], [])
    if not isinstance(gpus, list):
        gpus = []
    rows = []
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        total = float(gpu.get("memory_total_mib") or 0)
        used = float(gpu.get("memory_used_mib") or 0)
        util = float(gpu.get("utilization_gpu_percent") or 0)
        mem_pct = (used / total * 100.0) if total else 0.0
        rows.append({"name": gpu.get("name"), "utilization_gpu_percent": util, "memory_used_percent": round(mem_pct, 2)})
    busy = any(float(row.get("utilization_gpu_percent") or 0) >= 75 or float(row.get("memory_used_percent") or 0) >= 82 for row in rows)
    return {"busy": busy, "gpus": rows}


def local_disk_free() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    return {"free_gib": round(usage.free / 1024**3, 2), "path": str(ROOT)}


def service_status() -> dict[str, bool]:
    return {
        "dashboard": port_open("127.0.0.1", 8787),
        "hive_node": port_open("127.0.0.1", 8791),
        "hive_relay": port_open("127.0.0.1", 8793),
    }


def port_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def planned_spend_window(*, hours: int) -> dict[str, Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    total = 0.0
    event_count = 0
    for row in read_jsonl_tail(DEFAULT_LEDGER_PATH, 1000):
        try:
            created = datetime.fromisoformat(str(row.get("created_utc") or "").replace("Z", "+00:00"))
        except ValueError:
            continue
        if created < cutoff:
            continue
        if row.get("kind") in {"plan", "launch_dry_run", "launch_execute"}:
            value = row.get("estimated_total_usd")
            if isinstance(value, (int, float)):
                total += float(value)
                event_count += 1
    return {"window_hours": hours, "estimated_total_usd": round(total, 4), "event_count": event_count}


def launch_cooldown(cooldown_minutes: float) -> dict[str, Any]:
    launches = [row for row in read_jsonl_tail(DEFAULT_LEDGER_PATH, 200) if row.get("kind") in {"launch_execute", "launch_dry_run"}]
    if not launches:
        return {"passed": True, "cooldown_minutes": cooldown_minutes, "last_launch_utc": ""}
    last = launches[-1]
    try:
        created = datetime.fromisoformat(str(last.get("created_utc") or "").replace("Z", "+00:00"))
    except ValueError:
        return {"passed": True, "cooldown_minutes": cooldown_minutes, "last_launch_utc": last.get("created_utc")}
    elapsed = (datetime.now(timezone.utc) - created).total_seconds() / 60.0
    return {
        "passed": elapsed >= cooldown_minutes,
        "cooldown_minutes": cooldown_minutes,
        "elapsed_minutes": round(elapsed, 2),
        "last_launch_utc": last.get("created_utc"),
    }


def time_window_matches(window: dict[str, Any], dt: datetime) -> bool:
    days = [str(day).lower()[:3] for day in window.get("days", ["any"])]
    start = parse_hhmm(str(window.get("start_local") or "00:00"))
    end = parse_hhmm(str(window.get("end_local") or "23:59"))
    now_min = dt.hour * 60 + dt.minute
    today = dt.strftime("%a").lower()[:3]
    yesterday = (dt - timedelta(days=1)).strftime("%a").lower()[:3]
    def day_allowed(day: str) -> bool:
        return "any" in days or day in days
    if start <= end:
        return day_allowed(today) and start <= now_min <= end
    return (day_allowed(today) and now_min >= start) or (day_allowed(yesterday) and now_min <= end)


def parse_hhmm(value: str) -> int:
    match = re.match(r"^(\d{1,2}):(\d{2})$", value.strip())
    if not match:
        return 0
    hour = max(0, min(23, int(match.group(1))))
    minute = max(0, min(59, int(match.group(2))))
    return hour * 60 + minute


def cloud_commands(cloud: dict[str, Any], prefix: str) -> list[list[str]]:
    sequence_key = f"{prefix}_commands"
    single_key = f"{prefix}_command"
    rows = cloud.get(sequence_key)
    if isinstance(rows, list) and rows and all(isinstance(row, list) for row in rows):
        return [[str(item) for item in row] for row in rows]
    row = cloud.get(single_key)
    if isinstance(row, list) and row:
        return [[str(item) for item in row]]
    return []


def run_provider_commands(commands: list[list[str]], *, timeout: int) -> dict[str, Any]:
    results = []
    started = datetime.now(timezone.utc)
    for index, command in enumerate(commands):
        result = run_provider_command(command, timeout=timeout)
        result["index"] = index
        results.append(result)
        if result.get("returncode") != 0:
            break
    ok = bool(results) and all(row.get("returncode") == 0 for row in results)
    return {
        "returncode": 0 if ok else (results[-1].get("returncode") if results else 2),
        "ok": ok,
        "started_utc": started.isoformat(),
        "finished_utc": now(),
        "commands_run": len(results),
        "results": results,
        "stdout": "\n".join(str(row.get("stdout") or "") for row in results)[-4000:],
        "stderr": "\n".join(str(row.get("stderr") or "") for row in results)[-4000:],
    }


def run_provider_command(command: list[str], *, timeout: int) -> dict[str, Any]:
    if not command:
        return {"returncode": 2, "stdout": "", "stderr": "missing command"}
    expanded = expand_env_tokens(command)
    if not expanded.get("ok"):
        return {"returncode": 126, "stdout": "", "stderr": expanded.get("error", "env expansion failed")}
    command = expanded["command"]
    if not shutil.which(command[0]):
        return {"returncode": 127, "stdout": "", "stderr": f"{command[0]} not found"}
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": 124, "stdout": "", "stderr": str(exc)}


def expand_env_tokens(command: list[str]) -> dict[str, Any]:
    expanded: list[str] = []
    missing: list[str] = []
    pattern = re.compile(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}")
    for item in command:
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            value = os.environ.get(name)
            if value is None:
                missing.append(name)
                return ""
            return value
        expanded.append(pattern.sub(replace, str(item)))
    if missing:
        return {"ok": False, "error": "missing environment variable(s): " + ", ".join(sorted(set(missing)))}
    return {"ok": True, "command": expanded}


def public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "name",
        "provider",
        "enabled",
        "purpose",
        "region",
        "instance_type",
        "machine_type",
        "vm_size",
        "gpu_type",
        "root_volume_gib",
        "container_disk_gib",
        "disk_gib",
        "spot",
        "max_price_usd_per_hour",
        "max_session_hours",
        "allowed_task_kinds",
        "bucket",
        "project",
        "zone",
        "location",
        "resource_group",
        "conditions",
    ]
    return {key: redact_value(profile.get(key)) for key in keys if key in profile}


def compact_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict) or not plan:
        return {}
    return {
        "created_utc": plan.get("created_utc"),
        "decision": plan.get("decision"),
        "provider": plan.get("provider"),
        "profile": get_path(plan, ["profile", "name"], ""),
        "task_kind": plan.get("task_kind"),
        "estimated_total_usd": plan.get("estimated_total_usd"),
        "blockers": get_path(plan, ["conditions", "blockers"], []),
    }


def public_report(report: dict[str, Any]) -> dict[str, Any]:
    return redact_value(report)


def redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lower = str(key).lower()
            boolean_marker = any(marker in lower for marker in ("contains_secret", "contains_invite", "token_configured", "secret_configured"))
            if not boolean_marker and any(marker in lower for marker in ("token", "secret", "password", "access_key", "private_key")):
                out[key] = "***REDACTED***" if item else item
            else:
                out[key] = redact_value(item)
        return out
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, str) and ("theseus_hive_invite" in value or "base64 -d" in value):
        return "***REDACTED_BOOTSTRAP_WITH_INVITE***"
    return value


def redact_command(command: list[str]) -> list[str]:
    redacted = []
    skip_next = False
    for item in command:
        if skip_next:
            redacted.append("***REDACTED***")
            skip_next = False
            continue
        text = str(item)
        if "REDACTED_INVITE_B64" in text:
            redacted.append(text)
        elif "theseus_hive_invite" in text or "base64 -d" in text:
            redacted.append("***REDACTED_BOOTSTRAP_WITH_INVITE***")
        elif "{env:" in text or "Authorization: Bearer" in text:
            redacted.append(re.sub(r"\{env:[A-Za-z_][A-Za-z0-9_]*\}", "***ENV***", text))
        else:
            redacted.append(text)
        if item in {"--secret", "--token", "--password"}:
            skip_next = True
    return redacted


def safety_summary() -> dict[str, Any]:
    return {
        "dry_run_default": True,
        "explicit_execute_required": True,
        "public_benchmark_data_training_allowed": False,
        "arbitrary_shell_allowed": False,
        "cloud_credentials_stored_in_repo": False,
        "private_invites_only_in_ignored_local_config_or_user_data": True,
        "recommended_network": "private_subnet_or_private_relay; do not expose 8787/8791 publicly",
    }


def rented_compute_next_actions(profiles: list[dict[str, Any]], aws: dict[str, Any], last_plan: dict[str, Any]) -> list[str]:
    actions = []
    if not profiles:
        actions.append("theseus rent init --provider aws --name aws-gpu-nightly")
    if profiles and not aws.get("cli_installed"):
        actions.append("Install and configure the AWS CLI on the coordinator before executing AWS plans.")
    if profiles:
        actions.append(f"theseus rent plan --profile {profiles[0].get('name')} --task-kind cuda_training_chunk --hours 4")
    if last_plan.get("decision") == "rent":
        actions.append("theseus rent launch --plan reports/hive_rented_compute_plan.json --execute")
    return actions


def plan_next_actions(decision: str, blockers: list[str], cloud: dict[str, Any]) -> list[str]:
    if decision == "rent":
        return [
            "Review the launch command, security group, budget, AMI, and user-data path.",
            "python scripts/hive_rented_compute.py launch --plan reports/hive_rented_compute_plan.json --execute",
            "After the node joins, run: theseus hive training-link --refresh",
        ]
    actions = ["No rented capacity will be launched from this plan."]
    for blocker in blockers[:5]:
        actions.append(f"Resolve blocker: {blocker}")
    if cloud.get("adapter_maturity") == "R0_contract_only":
        actions.append("Implement this provider adapter or switch to aws_ec2/aws_s3.")
    return actions


def active_hive_public() -> dict[str, Any]:
    if hive_profiles is None:
        return redact_value(active_hive_config())
    try:
        active = hive_profiles.active_profile()
        return hive_profiles.public_profile(active) if active else {}
    except Exception:
        return redact_value(active_hive_config())


def active_hive_config() -> dict[str, Any]:
    policy = read_json(POLICY_PATH, {})
    join = read_json(ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json")), {})
    profiles = read_json(ROOT / str(get_path(policy, ["federation", "profiles_path"], "configs/hive_profiles.local.json")), {})
    active: dict[str, Any] = {}
    if isinstance(profiles, dict):
        active_id = str(profiles.get("active_profile_id") or "")
        for row in profiles.get("profiles") or []:
            if isinstance(row, dict) and row.get("profile_id") == active_id:
                active.update(row)
                break
    if isinstance(join, dict):
        active.update(join)
    return active


def default_repo_url() -> str:
    try:
        result = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, text=True, capture_output=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def default_branch() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True, capture_output=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def first_number(report: Any, names: list[str]) -> float | None:
    if isinstance(report, dict):
        for name in names:
            value = report.get(name)
            if isinstance(value, (int, float)):
                return float(value)
        for item in report.values():
            found = first_number(item, names)
            if found is not None:
                return found
    elif isinstance(report, list):
        for item in report:
            found = first_number(item, names)
            if found is not None:
                return found
    return None


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, details: dict[str, Any]) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def is_placeholder(value: str) -> bool:
    text = str(value or "").upper()
    return "REPLACE" in text or text in {"PROJECT", "ZONE", "RESOURCE_GROUP", "STORAGE_ACCOUNT", "OFFER_ID", "BUCKET"}


def safe_tag_key(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.:/=+\-@ ]+", "", str(value))[:120]
    return text or "TheseusTag"


def safe_tag_value(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:/=+\-@ ]+", "", str(value))[:240]


def safe_label_key(value: Any) -> str:
    text = re.sub(r"[^a-z0-9_-]+", "-", str(value).lower()).strip("-")
    return text[:63] or "theseus"


def safe_label_value(value: Any) -> str:
    text = re.sub(r"[^a-z0-9_-]+", "-", str(value).lower()).strip("-")
    return text[:63] or "value"


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    p = resolve_path(path)
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def read_json(path: str | Path, default: Any) -> Any:
    try:
        p = resolve_path(path)
        if not p.exists():
            return default
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: str | Path, data: Any) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_jsonl_tail(path: str | Path, limit: int) -> list[dict[str, Any]]:
    p = resolve_path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    except OSError:
        return []
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def ledger_event(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"created_utc": now(), "kind": kind, "node": socket.gethostname(), **payload}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_path(obj: Any, path: list[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


if __name__ == "__main__":
    raise SystemExit(main())
