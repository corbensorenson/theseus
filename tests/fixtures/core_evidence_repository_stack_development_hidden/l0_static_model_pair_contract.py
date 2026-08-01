from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "core_evidence_repository_stack_development.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("l0_runner_contract", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_fixed_worker_v3_is_the_default() -> None:
    assert runner.DEFAULT_WORKER_CONFIG.name == (
        "core_evidence_tmax_9b_worker_control_v3.json"
    ), "request_contract:fixed_worker_v3"


def test_l0_manifest_selects_only_l0_development_rows() -> None:
    selector = getattr(runner, "select_task_rows", None)
    assert callable(selector), "request_contract:l0_rows_only"
    tasks = [
        {
            "opaque_task_id": "eligible",
            "partition": "development",
            "denominator": "L0_DEVELOPMENT",
        },
        {
            "opaque_task_id": "wrong-denominator",
            "partition": "development",
            "denominator": "D1_DEVELOPMENT",
        },
        {
            "opaque_task_id": "wrong-partition",
            "partition": "qualification",
            "denominator": "L0_DEVELOPMENT",
        },
    ]
    selected = selector(
        tasks,
        manifest_policy="project_theseus_l0_real_work_task_manifest_v1",
        task_index=0,
        task_limit=10,
    )
    assert [row["opaque_task_id"] for row in selected] == [
        "eligible"
    ], "request_contract:l0_rows_only"


def test_l0_arm_aliases_reuse_target_blind_adapter_variants() -> None:
    mapper = getattr(runner, "canonical_arm_variant_id", None)
    assert callable(mapper), "request_contract:l0_arm_aliases"
    assert mapper("direct_fixed_worker") == "direct", (
        "request_contract:l0_arm_aliases"
    )
    assert mapper("full_theseus") == "full_stack", (
        "request_contract:l0_arm_aliases"
    )


def test_static_model_identity_and_config_drift_fail_closed() -> None:
    auditor = getattr(runner, "fixed_model_identity_audit", None)
    assert callable(auditor), "request_contract:static_model_identity"
    worker_config = json.loads(
        (
            ROOT
            / "configs"
            / "core_evidence_tmax_9b_worker_control_v3.json"
        ).read_text(encoding="utf-8")
    )
    config_sha = "fixed-config"

    def arm(arm_id: str, revision: str, runtime: str = "mlx") -> dict:
        return {
            "arm_id": arm_id,
            "dispatch_allowed": True,
            "candidate": {
                "model_identity": {
                    "repo_id": worker_config["model"]["repo_id"],
                    "revision": revision,
                    "runtime": runtime,
                },
                "candidate_seal": {"config_sha256": config_sha},
            },
        }

    valid = auditor(
        [
            arm("direct_fixed_worker", worker_config["model"]["revision"]),
            arm("full_theseus", worker_config["model"]["revision"]),
        ],
        worker_config=worker_config,
        worker_config_sha256=config_sha,
    )
    assert valid["passed"] is True, "request_contract:static_model_identity"

    revision_drift = auditor(
        [
            arm("direct_fixed_worker", worker_config["model"]["revision"]),
            arm("full_theseus", "different-revision"),
        ],
        worker_config=worker_config,
        worker_config_sha256=config_sha,
    )
    assert revision_drift["passed"] is False, (
        "request_contract:static_model_identity"
    )

    runtime_drift = auditor(
        [
            arm("direct_fixed_worker", worker_config["model"]["revision"]),
            arm(
                "full_theseus",
                worker_config["model"]["revision"],
                runtime="different-runtime",
            ),
        ],
        worker_config=worker_config,
        worker_config_sha256=config_sha,
    )
    assert runtime_drift["passed"] is False, (
        "request_contract:static_model_identity"
    )
