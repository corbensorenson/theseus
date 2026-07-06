"""Provider profile and catalog helpers for rented Hive compute/storage."""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime, timezone
from typing import Any


CONCRETE_PROVIDERS = {
    "aws_ec2",
    "aws_s3",
    "gcp_compute",
    "gcp_gcs",
    "azure_vm",
    "azure_blob",
    "runpod",
    "vast_ai",
}
PROVIDER_ALIASES = {
    "aws": "aws_ec2",
    "ec2": "aws_ec2",
    "s3": "aws_s3",
    "aws_s3_storage": "aws_s3",
    "gcp": "gcp_compute",
    "google": "gcp_compute",
    "gce": "gcp_compute",
    "gcs": "gcp_gcs",
    "google_storage": "gcp_gcs",
    "azure": "azure_vm",
    "az": "azure_vm",
    "azure_storage": "azure_blob",
    "blob": "azure_blob",
    "lambda": "lambda_cloud",
    "lambda_labs": "lambda_cloud",
    "vast": "vast_ai",
    "vastai": "vast_ai",
    "run_pod": "runpod",
}
PLANNING_PROVIDERS = (
    "lambda_cloud",
    "coreweave",
    "fluidstack",
    "ssh_gpu_host",
)
PROVIDER_CATALOG_ORDER = (
    "aws_ec2",
    "aws_s3",
    "gcp_compute",
    "gcp_gcs",
    "azure_vm",
    "azure_blob",
    "runpod",
    "vast_ai",
    *PLANNING_PROVIDERS,
)
DEFAULT_ALLOWED_TASK_KINDS = [
    "cuda_eval_chunk",
    "cuda_training_chunk",
    "cuda_rollout_chunk",
    "training_smoke",
]


def configured_profiles(policy: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in (get_path(policy, ["rented_compute", "profiles"], []), config.get("profiles") if isinstance(config, dict) else []):
        if isinstance(source, list):
            rows.extend([row for row in source if isinstance(row, dict)])
    return [normalize_profile(row) for row in rows]


def find_profile(policy: dict[str, Any], config: dict[str, Any], name: str) -> dict[str, Any] | None:
    wanted = safe_id(name)
    for profile in configured_profiles(policy, config):
        if safe_id(str(profile.get("name") or "")) == wanted:
            return profile
    return None


def normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    row = dict(profile)
    row["provider"] = canonical_provider(str(row.get("provider") or "aws_ec2"))
    row["name"] = safe_id(str(row.get("name") or default_profile_name(row["provider"])))
    if "allowed_task_kinds" not in row:
        row["allowed_task_kinds"] = list(DEFAULT_ALLOWED_TASK_KINDS)
    if "max_session_hours" not in row:
        row["max_session_hours"] = 24 if provider_kind(row["provider"]) == "storage" else 6
    return row


def default_compute_conditions() -> dict[str, Any]:
    return {
        "time_windows": [
            {"days": ["mon", "tue", "wed", "thu", "sun"], "start_local": "21:00", "end_local": "07:00"},
            {"days": ["fri", "sat"], "start_local": "00:00", "end_local": "23:59"},
        ],
        "spot_only": True,
        "max_estimated_hourly_usd": 0.75,
        "max_total_usd_per_session": 4.50,
        "max_daily_planned_usd": 12.0,
        "cooldown_minutes": 20,
        "require_queue_pressure": True,
        "require_broad_transfer_below_floor": True,
        "min_transfer_floor_gap": 0.03,
        "min_local_disk_free_gib": 10,
        "require_services": ["hive_node"],
        "stop_when_queue_empty": True,
        "stop_when_budget_spent": True,
    }


def default_tags(role: str) -> dict[str, str]:
    return {
        "Project": "ProjectTheseus",
        "HiveRole": role,
        "ManagedBy": "theseus-rent",
    }


def default_local_config(*, provider: str, name: str, region: str, repo_url: str, branch: str) -> dict[str, Any]:
    provider = canonical_provider(provider)
    base_conditions = default_compute_conditions()
    bootstrap = {
        "repo_url": repo_url or "REPLACE-WITH-PRIVATE-GIT-REPO",
        "branch": branch or "main",
        "join_from_active_hive": True,
        "include_join_token_in_user_data": True,
        "auto_update_soft": True,
        "public_contribution_mode": "off",
    }
    if provider == "aws_s3":
        profile = {
            "name": name,
            "provider": "aws_s3",
            "enabled": True,
            "region": region,
            "aws_profile": "",
            "bucket": "theseus-hive-REPLACE-WITH-UNIQUE-BUCKET",
            "policy_id": "storage_burst",
            "purpose": "Optional rented object storage for artifacts/checkpoint handoff. Upload policy remains controlled by Hive artifact sync.",
            "allowed_task_kinds": [
                "storage_burst",
                "artifact_sync",
                "checkpoint_handoff",
            ],
            "max_estimated_hourly_usd": 0.05,
            "max_total_usd_per_session": 2.0,
            "max_session_hours": 24,
            "conditions": {
                "spot_only": False,
                "require_queue_pressure": False,
                "require_broad_transfer_below_floor": False,
                "require_services": [],
                "max_total_usd_per_session": 2.0,
                "min_local_disk_free_gib": 5,
            },
        }
    elif provider == "gcp_gcs":
        profile = {
            "name": name,
            "provider": "gcp_gcs",
            "enabled": True,
            "region": region,
            "project": "REPLACE-WITH-GCP-PROJECT",
            "bucket": "theseus-hive-REPLACE-WITH-UNIQUE-BUCKET",
            "location": "US",
            "policy_id": "storage_burst",
            "purpose": "Optional rented object storage for artifacts/checkpoint handoff. Upload policy remains controlled by Hive artifact sync.",
            "allowed_task_kinds": ["storage_burst", "artifact_sync", "checkpoint_handoff"],
            "max_estimated_hourly_usd": 0.05,
            "max_total_usd_per_session": 2.0,
            "max_session_hours": 24,
            "conditions": {
                "spot_only": False,
                "require_queue_pressure": False,
                "require_broad_transfer_below_floor": False,
                "require_services": [],
                "max_total_usd_per_session": 2.0,
                "min_local_disk_free_gib": 5,
            },
        }
    elif provider == "azure_blob":
        profile = {
            "name": name,
            "provider": "azure_blob",
            "enabled": True,
            "region": region,
            "resource_group": "REPLACE-WITH-RESOURCE-GROUP",
            "storage_account": "theseushiveREPLACE",
            "container": "theseus-hive",
            "location": region,
            "sku": "Standard_LRS",
            "policy_id": "storage_burst",
            "purpose": "Optional rented Azure Blob storage for artifacts/checkpoint handoff. Upload policy remains controlled by Hive artifact sync.",
            "allowed_task_kinds": ["storage_burst", "artifact_sync", "checkpoint_handoff"],
            "max_estimated_hourly_usd": 0.05,
            "max_total_usd_per_session": 2.0,
            "max_session_hours": 24,
            "conditions": {
                "spot_only": False,
                "require_queue_pressure": False,
                "require_broad_transfer_below_floor": False,
                "require_services": [],
                "max_total_usd_per_session": 2.0,
                "min_local_disk_free_gib": 5,
            },
        }
    elif provider == "gcp_compute":
        profile = {
            "name": name,
            "provider": provider,
            "enabled": True,
            "purpose": "Night/overflow rented GCP GPU worker for transferable private training/eval chunks.",
            "project": "REPLACE-WITH-GCP-PROJECT",
            "zone": "us-central1-a",
            "machine_type": "g2-standard-4",
            "policy_id": "overnight_spot_gpu",
            "image_family": "ubuntu-2204-lts",
            "image_project": "ubuntu-os-cloud",
            "network": "",
            "subnet": "",
            "no_external_ip": True,
            "accelerator": {"type": "nvidia-l4", "count": 1},
            "root_volume_gib": 250,
            "spot": True,
            "max_price_usd_per_hour": 0.75,
            "max_session_hours": 6,
            "allowed_task_kinds": list(DEFAULT_ALLOWED_TASK_KINDS),
            "conditions": base_conditions,
            "bootstrap": bootstrap,
            "tags": default_tags("gcp-worker"),
        }
    elif provider == "azure_vm":
        profile = {
            "name": name,
            "provider": provider,
            "enabled": True,
            "purpose": "Night/overflow rented Azure GPU worker for transferable private training/eval chunks.",
            "resource_group": "REPLACE-WITH-RESOURCE-GROUP",
            "location": region,
            "vm_name": name,
            "vm_size": "Standard_NC4as_T4_v3",
            "policy_id": "overnight_spot_gpu",
            "image": "Ubuntu2204",
            "admin_username": "theseus",
            "vnet_name": "",
            "subnet": "",
            "root_volume_gib": 250,
            "spot": True,
            "eviction_policy": "Deallocate",
            "max_price_usd_per_hour": 0.75,
            "max_session_hours": 6,
            "allowed_task_kinds": list(DEFAULT_ALLOWED_TASK_KINDS),
            "conditions": base_conditions,
            "bootstrap": bootstrap,
            "tags": default_tags("azure-worker"),
        }
    elif provider == "runpod":
        profile = {
            "name": name,
            "provider": provider,
            "enabled": True,
            "purpose": "Reviewed rented RunPod GPU pod for overflow private training/eval chunks.",
            "gpu_type": "NVIDIA GeForce RTX 4090",
            "gpu_count": 1,
            "image": "pytorch/pytorch:latest",
            "container_disk_gib": 120,
            "cloud_type": "SECURE",
            "global_networking": False,
            "spot": True,
            "policy_id": "overnight_spot_gpu",
            "max_price_usd_per_hour": 0.75,
            "max_session_hours": 4,
            "allowed_task_kinds": list(DEFAULT_ALLOWED_TASK_KINDS),
            "conditions": base_conditions,
            "bootstrap": bootstrap,
            "tags": default_tags("runpod-worker"),
        }
    elif provider == "vast_ai":
        profile = {
            "name": name,
            "provider": provider,
            "enabled": True,
            "purpose": "Reviewed rented Vast.ai GPU instance for overflow private training/eval chunks.",
            "offer_id": "REPLACE-WITH-VAST-OFFER-ID",
            "image": "pytorch/pytorch:latest",
            "disk_gib": 120,
            "spot": True,
            "policy_id": "overnight_spot_gpu",
            "bid_price": 0.30,
            "max_price_usd_per_hour": 0.75,
            "max_session_hours": 4,
            "allowed_task_kinds": list(DEFAULT_ALLOWED_TASK_KINDS),
            "conditions": base_conditions,
            "bootstrap": bootstrap,
            "tags": default_tags("vast-worker"),
        }
    else:
        profile = {
            "name": name,
            "provider": provider,
            "enabled": True,
            "purpose": "Night/overflow rented Hive worker for transferable private training/eval chunks.",
            "region": region,
            "policy_id": "overnight_spot_gpu",
            "aws_profile": "",
            "ami_id": "ami-REPLACE-WITH-UBUNTU-OR-DEEP-LEARNING-AMI",
            "instance_type": "g5.xlarge",
            "key_name": "",
            "subnet_id": "",
            "security_group_ids": [],
            "iam_instance_profile": "",
            "allow_default_vpc": falsey_env("THESEUS_RENT_REQUIRE_EXPLICIT_SUBNET"),
            "allow_default_security_group": False,
            "root_volume_gib": 250,
            "spot": True,
            "max_price_usd_per_hour": 0.75,
            "max_session_hours": 6,
            "allowed_task_kinds": list(DEFAULT_ALLOWED_TASK_KINDS),
            "conditions": base_conditions,
            "bootstrap": bootstrap,
            "tags": default_tags("rented-worker"),
        }
    return {
        "policy": "project_theseus_hive_rented_compute_local_config_v0",
        "created_utc": now(),
        "profiles": [profile],
    }


def provider_catalog(providers: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for provider in PROVIDER_CATALOG_ORDER:
        status = providers.get(provider, {})
        rows.append(
            {
                "provider": provider,
                "kind": provider_kind(provider),
                "maturity": provider_maturity(provider),
                "implemented": provider in CONCRETE_PROVIDERS,
                "cli_ready": bool(status.get("cli_installed")),
                "credential_hint": status.get("credentials_hint", ""),
                "notes": provider_notes(provider),
            }
        )
    return rows


def provider_status() -> dict[str, dict[str, Any]]:
    curl = shutil.which("curl")
    return {
        "aws_ec2": aws_status(),
        "aws_s3": aws_status(),
        "gcp_compute": cli_status("gcloud", credentials_hint="run_gcloud_auth_login_or_use_service_account"),
        "gcp_gcs": cli_status("gcloud", credentials_hint="run_gcloud_auth_login_or_use_service_account"),
        "azure_vm": cli_status("az", credentials_hint="run_az_login_or_configure_service_principal"),
        "azure_blob": cli_status("az", credentials_hint="run_az_login_or_configure_service_principal"),
        "runpod": {"cli_installed": bool(curl), "cli_path": curl or "", "credentials_hint": "set_RUNPOD_API_KEY"},
        "vast_ai": cli_status("vastai", credentials_hint="run_vastai_set_api_key_or_set_config"),
        "lambda_cloud": cli_status("lambda", credentials_hint="set_lambda_cloud_credentials_when_adapter_is_promoted"),
        "coreweave": cli_status("kubectl", credentials_hint="configure_kube_context_when_adapter_is_promoted"),
        "fluidstack": cli_status("fluidstack", credentials_hint="set_fluidstack_credentials_when_adapter_is_promoted"),
        "ssh_gpu_host": cli_status("ssh", credentials_hint="configure_ssh_key_and_host"),
    }


def cli_status(executable: str, *, credentials_hint: str) -> dict[str, Any]:
    path = shutil.which(executable)
    return {
        "cli_installed": bool(path),
        "cli_path": path or "",
        "credentials_hint": credentials_hint,
    }


def provider_kind(provider: str) -> str:
    if provider in {"aws_s3", "gcp_gcs", "azure_blob"}:
        return "storage"
    return "compute"


def default_task_kind(provider: str) -> str:
    return "storage_burst" if provider_kind(provider) == "storage" else "cuda_training_chunk"


def default_plan_hours(provider: str) -> int:
    return 1 if provider_kind(provider) == "storage" else 4


def provider_maturity(provider: str) -> str:
    if provider in {"aws_ec2", "gcp_compute", "azure_vm"}:
        return "R2"
    if provider in {"aws_s3", "gcp_gcs", "azure_blob", "runpod", "vast_ai"}:
        return "R1"
    return "R0"


def provider_notes(provider: str) -> str:
    notes = {
        "aws_ec2": "Launches one EC2 worker from an explicit dry-run plan.",
        "aws_s3": "Creates an S3 bucket only; Hive artifact sync controls upload policy.",
        "gcp_compute": "Launches one Compute Engine worker with a startup script when gcloud is configured.",
        "gcp_gcs": "Creates a Google Cloud Storage bucket only.",
        "azure_vm": "Launches one Azure Linux VM with cloud-init custom data when az is configured.",
        "azure_blob": "Creates an Azure storage account/container pair only.",
        "runpod": "Creates a reviewed RunPod GPU pod through the REST API using RUNPOD_API_KEY.",
        "vast_ai": "Creates a reviewed Vast.ai instance from a known offer id using the vastai CLI.",
        "lambda_cloud": "Profile/condition contract only until Lambda Cloud adapter is promoted.",
        "coreweave": "Profile/condition contract only until Kubernetes manifest adapter is promoted.",
        "fluidstack": "Profile/condition contract only until Fluidstack adapter is promoted.",
        "ssh_gpu_host": "Profile/condition contract for manually rented hosts reachable by SSH.",
    }
    return notes.get(provider, "Provider profile and conditions are supported.")


def aws_status() -> dict[str, Any]:
    aws_path = shutil.which("aws")
    env_creds = bool(os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID"))
    return {
        "cli_installed": bool(aws_path),
        "cli_path": aws_path or "",
        "env_credentials_present": env_creds,
        "credentials_hint": "AWS_PROFILE_or_AWS_ACCESS_KEY_ID_present" if env_creds else "run_aws_configure_or_set_AWS_PROFILE",
    }


def provider_required_fields(provider: str) -> list[str]:
    if provider == "gcp_compute":
        return ["project", "zone", "machine_type", "image_family", "image_project", "accelerator"]
    if provider == "gcp_gcs":
        return ["project", "bucket", "location"]
    if provider == "azure_vm":
        return ["resource_group", "location", "vm_size", "image", "admin_username"]
    if provider == "azure_blob":
        return ["resource_group", "location", "storage_account", "container"]
    if provider == "runpod":
        return ["RUNPOD_API_KEY", "gpu_type", "image", "container_disk_gib"]
    if provider == "vast_ai":
        return ["api_key_env", "offer_filters", "image", "disk_gib"]
    if provider == "lambda_cloud":
        return ["api_key_env", "region", "instance_type", "ssh_key"]
    return ["provider_specific_fields"]


def default_profile_name(provider: str) -> str:
    provider = canonical_provider(provider)
    if provider == "aws_s3":
        return "aws-storage-burst"
    if provider == "gcp_gcs":
        return "gcp-storage-burst"
    if provider == "azure_blob":
        return "azure-storage-burst"
    if provider == "gcp_compute":
        return "gcp-gpu-nightly"
    if provider == "azure_vm":
        return "azure-gpu-nightly"
    if provider == "runpod":
        return "runpod-gpu-burst"
    if provider == "vast_ai":
        return "vast-gpu-burst"
    return "aws-gpu-nightly"


def provider_default_region(provider: str, requested: str) -> str:
    provider = canonical_provider(provider)
    if requested and requested != "us-east-1":
        return requested
    if provider in {"azure_vm", "azure_blob"}:
        return "eastus"
    if provider in {"gcp_gcs"}:
        return "US"
    return requested or "us-east-1"


def canonical_provider(provider: str) -> str:
    value = safe_id(provider or "aws_ec2").replace("-", "_")
    return PROVIDER_ALIASES.get(value, value)


def profile_hourly_estimate(profile: dict[str, Any]) -> float:
    for key in ("estimated_hourly_usd", "max_price_usd_per_hour", "max_estimated_hourly_usd"):
        try:
            value = float(profile.get(key) or 0)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return 0.0


def safe_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-._").lower()
    return slug or "profile"


def falsey_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"0", "false", "no", "off"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_path(obj: Any, path: list[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
