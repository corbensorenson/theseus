"""Real-training preflight for Project Theseus.

This gate keeps long training from starting until the ratchet, CUDA path,
public/private benchmarks, and promotion rules are visible and measurable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="configs/training_profiles_rtx2060super.json")
    parser.add_argument("--ablation-matrix", default="configs/ablation_matrix_rtx2060super.json")
    parser.add_argument("--architecture-gate", default="reports/architecture_gate_report.json")
    parser.add_argument("--external-inference-audit", default="reports/external_inference_audit.json")
    parser.add_argument("--rmi", default="reports/ratcheting_modular_intelligence_report.json")
    parser.add_argument("--split-leakage", default="reports/babylm_split_leakage_report.json")
    parser.add_argument("--candidate-gate", default="reports/candidate_promotion_gate.json")
    parser.add_argument("--frontier-report", default="")
    parser.add_argument("--candidate-gate-profile-step-in-progress", default="")
    parser.add_argument("--standalone-smoke", default="reports/preflight_cuda_standalone_smoke.json")
    parser.add_argument("--rollout-smoke", default="reports/preflight_cuda_rollout_smoke.json")
    parser.add_argument("--out", default="reports/training_preflight_report.json")
    parser.add_argument("--run-build-check", action="store_true")
    parser.add_argument("--run-smokes", action="store_true")
    parser.add_argument("--run-split-check", action="store_true")
    parser.add_argument("--run-candidate-gate", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    profiles = read_json(Path(args.profiles))
    commands = []
    if args.run_build_check:
        commands.append(run_command(["cargo", "check", "-p", "symliquid-cuda", "--features", "cuda"]))
        commands.append(
            run_command(
                ["cargo", "build", "--release", "-p", "symliquid-cli", "--features", "cuda"],
                timeout=600,
            )
        )
    if args.run_smokes:
        commands.extend(run_smoke_commands(profiles, args))
    if args.run_split_check:
        commands.append(run_command([sys.executable, "scripts/check_babylm_splits.py", "--out", args.split_leakage]))
    if args.run_candidate_gate:
        candidate_gate_command = [
            sys.executable,
            "scripts/candidate_promotion_gate.py",
            "--runtime-report",
            args.rollout_smoke,
            "--out",
            args.candidate_gate,
        ]
        if args.frontier_report:
            candidate_gate_command.extend(["--frontier-report", args.frontier_report])
        if args.candidate_gate_profile_step_in_progress:
            candidate_gate_command.extend(
                [
                    "--allow-profile-step-in-progress",
                    args.candidate_gate_profile_step_in_progress,
                ]
            )
        commands.append(
            run_command(
                candidate_gate_command,
                allow_failure=True,
            )
        )

    architecture = read_json(Path(args.architecture_gate))
    external_audit = read_json(Path(args.external_inference_audit))
    rmi = read_json(Path(args.rmi))
    split_leakage = read_json(Path(args.split_leakage))
    candidate_gate = read_json(Path(args.candidate_gate))
    standalone_smoke = read_json(Path(args.standalone_smoke))
    rollout_smoke = read_json(Path(args.rollout_smoke))
    ablation = read_json(Path(args.ablation_matrix))

    environment = collect_environment()
    checks = build_checks(
        profiles=profiles,
        ablation=ablation,
        environment=environment,
        architecture=architecture,
        external_audit=external_audit,
        rmi=rmi,
        split_leakage=split_leakage,
        candidate_gate=candidate_gate,
        standalone_smoke=standalone_smoke,
        rollout_smoke=rollout_smoke,
    )
    passed = sum(1 for item in checks if item["passed"])
    blockers = [item for item in checks if not item["passed"] and item["severity"] == "blocker"]
    warnings = [item for item in checks if not item["passed"] and item["severity"] == "warning"]
    report = {
        "policy": "local_only_no_external_inference",
        "methodology": "rmi_real_training_preflight",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_target": hardware_target_label(profiles, environment),
        "heavy_training_allowed": not blockers,
        "passed": passed,
        "total": len(checks),
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
        "environment": environment,
        "profiles": {
            "path": args.profiles,
            "profile_names": sorted((profiles.get("profiles") or {}).keys()),
        },
        "ablation_matrix": {
            "path": args.ablation_matrix,
            "comparison_count": len(ablation.get("comparisons", [])),
            "policy": ablation.get("policy"),
        },
        "commands": commands,
        "next_actions": next_actions(blockers, warnings),
        "artifacts": {
            "architecture_gate": args.architecture_gate,
            "external_inference_audit": args.external_inference_audit,
            "rmi": args.rmi,
            "split_leakage": args.split_leakage,
            "candidate_gate": args.candidate_gate,
            "standalone_smoke": args.standalone_smoke,
            "rollout_smoke": args.rollout_smoke,
        },
    }
    write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))
    return 1 if args.strict and blockers else 0


def run_smoke_commands(profiles: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    smoke = (((profiles.get("profiles") or {}).get("smoke") or {}))
    standalone = smoke.get("cgs_cuda_readout") or {}
    rollout = smoke.get("puffer_ocean_rollout_cuda") or {}
    exe = str(ROOT / "target" / "release" / "symliquid-cli.exe")
    return [
        run_command(
            [
                exe,
                "train-standalone-cuda",
                "--cases-per-task",
                str(standalone.get("cases_per_task", 4)),
                "--epochs",
                str(standalone.get("epochs", 1)),
                "--samples-per-launch",
                str(standalone.get("samples_per_launch", 64)),
                "--hv-dim",
                str(standalone.get("hv_dim", 512)),
                "--lr",
                str(standalone.get("lr", 0.05)),
                "--out",
                args.standalone_smoke,
            ],
            timeout=180,
        ),
        run_command(
            [
                exe,
                "train-rollout-cuda",
                "--cases-per-task",
                str(rollout.get("cases_per_task", 4)),
                "--epochs",
                str(rollout.get("epochs", 1)),
                "--state-epochs",
                str(rollout.get("state_epochs", 0)),
                "--state-lr",
                str(rollout.get("state_lr", 0.0)),
                "--probe-cases-per-task",
                str(rollout.get("probe_cases_per_task", 4)),
                "--samples-per-launch",
                str(rollout.get("samples_per_launch", 64)),
                "--rollout-batch",
                str(rollout.get("rollout_batch", 32)),
                "--obs-dim",
                str(rollout.get("obs_dim", 16)),
                "--hidden-dim",
                str(rollout.get("hidden_dim", 24)),
                "--reservoir-dim",
                str(rollout.get("reservoir_dim", 32)),
                "--hv-dim",
                str(rollout.get("hv_dim", 256)),
                "--seq-len",
                str(rollout.get("seq_len", 16)),
                "--lr",
                str(rollout.get("lr", 0.03)),
                "--out",
                args.rollout_smoke,
            ],
            timeout=180,
        ),
    ]


def build_checks(
    *,
    profiles: dict[str, Any],
    ablation: dict[str, Any],
    environment: dict[str, Any],
    architecture: dict[str, Any],
    external_audit: dict[str, Any],
    rmi: dict[str, Any],
    split_leakage: dict[str, Any],
    candidate_gate: dict[str, Any],
    standalone_smoke: dict[str, Any],
    rollout_smoke: dict[str, Any],
) -> list[dict[str, Any]]:
    smoke_gates = (
        ((profiles.get("profiles") or {}).get("smoke") or {}).get("promotion_gates") or {}
    )
    rollout_runtime_max = int(smoke_gates.get("rollout_total_runtime_ms_max", 120000))
    rollout_eps_min = float(smoke_gates.get("rollout_examples_per_second_min", 10.0))
    standalone_eps_min = float(smoke_gates.get("standalone_examples_per_second_min", 50.0))
    seed55_exists = Path("reports/babylm_mutated_holdout_seed55_stateful_grammar_state_frontier.json").exists()
    return [
        gate("profiles_defined", bool((profiles.get("profiles") or {}).keys()), "blocker", f"profiles={sorted((profiles.get('profiles') or {}).keys())}"),
        gate("ablation_matrix_defined", len(ablation.get("comparisons", [])) >= 8, "blocker", f"comparisons={len(ablation.get('comparisons', []))}"),
        gate(
            "gpu_matches_training_profile",
            gpu_matches_profile(environment.get("gpu_name"), profiles),
            "warning",
            f"gpu={environment.get('gpu_name')} expected={get_path(profiles, ['hardware', 'name'], '')}",
        ),
        gate("cuda_toolkit_present", bool(environment.get("cuda_toolkit")), "blocker", f"cuda_toolkit={environment.get('cuda_toolkit')}"),
        gate("rust_present", bool(environment.get("rustc")), "blocker", f"rustc={environment.get('rustc')}"),
        gate("release_binary_present", Path("target/release/symliquid-cli.exe").exists(), "blocker", "target/release/symliquid-cli.exe"),
        gate(
            "msvc_available",
            bool(environment.get("cl_visible")) or bool(environment.get("vsdevcmd_cl_visible")),
            "warning",
            f"cl_visible={environment.get('cl_visible')} vsdevcmd_cl_visible={environment.get('vsdevcmd_cl_visible')}",
        ),
        gate(
            "puffer_torch_cuda_state_recorded",
            environment.get("puffer_torch_cuda_available") is not None,
            "warning",
            f"puffer_torch_cuda={environment.get('puffer_torch_cuda_available')}",
        ),
        gate("puffer_compiled_backend_recorded", environment.get("puffer_compiled_backend_available") is not None, "warning", f"puffer_compiled_backend={environment.get('puffer_compiled_backend_available')}"),
        gate("architecture_gate_green", bool(architecture.get("ready_for_heavy_training")), "blocker", f"ready={architecture.get('ready_for_heavy_training')}"),
        gate(
            "external_inference_teacher_only",
            bool(external_audit.get("ok")) and external_audit.get("teacher_only_invariant") is True,
            "blocker",
            f"ok={external_audit.get('ok')} summary={external_audit.get('summary')}",
        ),
        gate("rmi_score_green", get_path(rmi, ["implementation_score", "score"], 0.0) >= 1.0, "blocker", f"score={get_path(rmi, ['implementation_score', 'score'], None)}"),
        gate("split_leakage_clean", bool(split_leakage.get("ok")), "blocker", f"ok={split_leakage.get('ok')} pair_overlaps={split_leakage.get('total_pair_overlaps')}"),
        gate("split_strict_quality", bool(split_leakage.get("strict_ok")), "warning", f"strict_ok={split_leakage.get('strict_ok')} sentence_overlaps={split_leakage.get('total_sentence_overlaps')}"),
        gate("standalone_cuda_smoke_present", bool(standalone_smoke), "blocker", "reports/preflight_cuda_standalone_smoke.json"),
        gate("standalone_cuda_no_fallback", standalone_smoke.get("cuda_fallback") is False, "blocker", f"cuda_fallback={standalone_smoke.get('cuda_fallback')}"),
        gate("standalone_cuda_fast_enough", float(standalone_smoke.get("train_examples_per_second") or 0.0) >= standalone_eps_min, "blocker", f"eps={standalone_smoke.get('train_examples_per_second')} min={standalone_eps_min}"),
        gate("rollout_cuda_smoke_present", bool(rollout_smoke), "blocker", "reports/preflight_cuda_rollout_smoke.json"),
        gate("rollout_cuda_no_fallback", rollout_smoke.get("cuda_fallback") is False, "blocker", f"cuda_fallback={rollout_smoke.get('cuda_fallback')}"),
        gate("rollout_cuda_telemetry_present", bool(rollout_smoke.get("runtime_profile")) and bool(rollout_smoke.get("timing_breakdown_ms")), "blocker", f"runtime={bool(rollout_smoke.get('runtime_profile'))} timing={bool(rollout_smoke.get('timing_breakdown_ms'))}"),
        gate("rollout_cuda_fast_enough", float(rollout_smoke.get("train_examples_per_second") or 0.0) >= rollout_eps_min, "blocker", f"eps={rollout_smoke.get('train_examples_per_second')} min={rollout_eps_min}"),
        gate("rollout_cuda_runtime_bounded", int(rollout_smoke.get("train_runtime_ms") or 10**12) <= rollout_runtime_max, "blocker", f"runtime_ms={rollout_smoke.get('train_runtime_ms')} max={rollout_runtime_max}"),
        gate("candidate_gate_recorded", bool(candidate_gate), "warning", "reports/candidate_promotion_gate.json"),
        gate(
            "candidate_promotion_state_consistent",
            seed55_exists or candidate_gate.get("promote") is not True,
            "warning",
            f"seed55_exists={seed55_exists} promote={candidate_gate.get('promote')}",
        ),
    ]


def gate(name: str, passed: bool, severity: str, evidence: str) -> dict[str, Any]:
    return {
        "gate": name,
        "passed": bool(passed),
        "severity": severity,
        "evidence": evidence,
    }


def gpu_matches_profile(gpu_name: Any, profiles: dict[str, Any]) -> bool:
    expected = str(get_path(profiles, ["hardware", "name"], "") or "")
    observed = str(gpu_name or "")
    if not expected or not observed:
        return False
    return normalize_gpu_name(expected) in normalize_gpu_name(observed)


def hardware_target_label(profiles: dict[str, Any], environment: dict[str, Any]) -> str:
    hardware = profiles.get("hardware") if isinstance(profiles.get("hardware"), dict) else {}
    name = str(hardware.get("name") or environment.get("gpu_name") or "unknown_gpu")
    compute = str(hardware.get("compute_capability") or environment.get("compute_capability") or "")
    return f"{normalize_gpu_name(name)}_sm{compute.replace('.', '') or 'unknown'}"


def normalize_gpu_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {}
    smi = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        allow_failure=True,
    )
    if smi["returncode"] == 0 and smi["stdout"].strip():
        parts = [part.strip() for part in smi["stdout"].splitlines()[0].split(",")]
        keys = [
            "gpu_name",
            "vram_total_mib",
            "vram_free_mib",
            "driver_version",
            "compute_capability",
            "gpu_utilization_percent",
        ]
        env.update({key: parts[idx] for idx, key in enumerate(keys) if idx < len(parts)})
    env["nvidia_smi"] = smi
    env["nvcc"] = run_command(["nvcc", "--version"], allow_failure=True)
    env["cuda_toolkit"] = extract_cuda_release(env["nvcc"].get("stdout", ""))
    rustc_cmd = resolve_tool("rustc")
    cargo_cmd = resolve_tool("cargo")
    env["rustc_path"] = rustc_cmd[0]
    env["cargo_path"] = cargo_cmd[0]
    env["rustc"] = single_line(run_command([*rustc_cmd, "--version"], allow_failure=True).get("stdout"))
    env["cargo"] = single_line(run_command([*cargo_cmd, "--version"], allow_failure=True).get("stdout"))
    cl = run_command(["where.exe", "cl"], allow_failure=True)
    env["cl_visible"] = cl["returncode"] == 0
    env["cl_where"] = cl.get("stdout", "").strip()
    env.update(check_msvc_dev_shell())
    env.update(check_puffer_stack())
    return env


def check_msvc_dev_shell() -> dict[str, Any]:
    vswhere = Path(os_environ_program_files_x86()) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return {"vswhere": None, "vsdevcmd": None, "vsdevcmd_cl_visible": False}
    install = run_command(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        allow_failure=True,
    )
    path = single_line(install.get("stdout"))
    if not path:
        return {"vswhere": str(vswhere), "vsdevcmd": None, "vsdevcmd_cl_visible": False}
    devcmd = Path(path) / "Common7" / "Tools" / "VsDevCmd.bat"
    if not devcmd.exists():
        return {"vswhere": str(vswhere), "vsdevcmd": str(devcmd), "vsdevcmd_cl_visible": False}
    probe = run_shell_command(
        f'cmd.exe /s /c ""{devcmd}" -arch=x64 -host_arch=x64 >nul && where cl"',
    )
    return {
        "vswhere": str(vswhere),
        "vsdevcmd": str(devcmd),
        "vsdevcmd_cl_visible": probe["returncode"] == 0,
        "vsdevcmd_cl_where": probe.get("stdout", "").strip(),
    }


def os_environ_program_files_x86() -> str:
    import os

    return os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")


def check_puffer_stack() -> dict[str, Any]:
    python = ROOT / ".venv-puffer" / "Scripts" / "python.exe"
    if not python.exists():
        return {
            "puffer_python": None,
            "puffer_torch_cuda_available": None,
            "puffer_compiled_backend_available": None,
        }
    code = (
        "import json\n"
        "out={}\n"
        "try:\n"
        " import torch; out['torch_version']=torch.__version__; out['torch_cuda_available']=bool(torch.cuda.is_available())\n"
        "except Exception as e: out['torch_error']=str(e); out['torch_cuda_available']=None\n"
        "try:\n"
        " import pufferlib; out['pufferlib_version']=getattr(pufferlib,'__version__','unknown')\n"
        "except Exception as e: out['pufferlib_error']=str(e)\n"
        "try:\n"
        " import pufferlib._C; out['compiled_backend_available']=True\n"
        "except Exception as e: out['compiled_backend_available']=False; out['compiled_backend_error']=str(e)\n"
        "print(json.dumps(out))\n"
    )
    result = run_command([str(python), "-c", code], allow_failure=True)
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(result.get("stdout", "{}"))
    except json.JSONDecodeError:
        payload = {"puffer_probe_error": result.get("stderr")}
    return {
        "puffer_python": str(python),
        "puffer_torch_version": payload.get("torch_version"),
        "puffer_torch_cuda_available": payload.get("torch_cuda_available"),
        "puffer_compiled_backend_available": payload.get("compiled_backend_available"),
        "puffer_compiled_backend_error": payload.get("compiled_backend_error"),
    }


def next_actions(blockers: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> list[str]:
    names = {item["gate"] for item in blockers + warnings}
    actions = []
    if "rollout_cuda_fast_enough" in names or "rollout_cuda_runtime_bounded" in names:
        actions.append("continue_cuda_rollout_hot_path_work_before_long_training")
    if "external_inference_teacher_only" in names:
        actions.append("remove_or_route_non_teacher_external_inference_before_training")
    if "split_leakage_clean" in names:
        actions.append("generate_seed55_holdout_and_rerun_split_leakage_check")
    if "standalone_cuda_smoke_present" in names or "rollout_cuda_smoke_present" in names:
        actions.append("run_training_preflight_with_build_and_smokes")
    if "rust_present" in names:
        actions.append("add_cargo_home_bin_to_shell_path_or_continue_using_explicit_cargo_path")
    if "msvc_available" in names:
        actions.append("use_visual_studio_developer_shell_before_puffer_native_extension_work")
    if not actions:
        actions.append("run_inner_loop_profile_then_candidate_gate")
    return actions


def resolve_tool(name: str) -> list[str]:
    probe = run_command(["where.exe", name], allow_failure=True)
    if probe["returncode"] == 0:
        first = probe.get("stdout", "").splitlines()[0].strip()
        if first:
            return [first]
    cargo_home = Path.home() / ".cargo" / "bin" / f"{name}.exe"
    if cargo_home.exists():
        return [str(cargo_home)]
    return [name]


def run_command(
    command: list[str],
    *,
    timeout: int = 120,
    allow_failure: bool = False,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        out = {
            "command": command,
            "returncode": result.returncode,
            "runtime_ms": int((time.perf_counter() - start) * 1000),
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }
    except Exception as error:
        out = {
            "command": command,
            "returncode": -1,
            "runtime_ms": int((time.perf_counter() - start) * 1000),
            "stdout": "",
            "stderr": str(error),
        }
    if out["returncode"] != 0 and not allow_failure:
        return out
    return out


def run_shell_command(command: str, *, timeout: int = 120) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            shell=True,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "runtime_ms": int((time.perf_counter() - start) * 1000),
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }
    except Exception as error:
        return {
            "command": command,
            "returncode": -1,
            "runtime_ms": int((time.perf_counter() - start) * 1000),
            "stdout": "",
            "stderr": str(error),
        }


def extract_cuda_release(text: str) -> str | None:
    for line in text.splitlines():
        if "release" in line:
            return line.strip()
    return None


def single_line(text: Any) -> str | None:
    if not text:
        return None
    return str(text).strip().splitlines()[0]


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
