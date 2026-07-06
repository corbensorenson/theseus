"""Probe local Minecraft/Open-World RL readiness without launching gameplay.

The probe is intentionally conservative. It detects local assets, Python
modules, Java, and source cards, but it does not store credentials, download
commercial content, join public servers, or use external inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "minecraft_rl_policy.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default="reports/minecraft_runtime_probe.json")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    install_paths = detect_install_paths(policy)
    java = command_version(["java", "-version"])
    node = command_version(["node", "--version"])
    modules = module_status()
    source_cards = source_card_status()

    full_runtime_ready = bool(
        install_paths
        and java.get("available")
        and (
            modules["minedojo"]["available"]
            or modules["malmoenv"]["available"]
            or modules["MalmoPython"]["available"]
            or modules["minerl"]["available"]
        )
    )
    bridge_runtime_ready = bool(modules["crafter"]["available"] or modules["craftax"]["available"])
    policy_ready = bool(policy.get("user_license", {}).get("required")) and bool(
        policy.get("user_license", {}).get("user_reported_license_for_this_machine")
    )
    report = {
        "policy": "project_theseus_minecraft_runtime_probe_v0",
        "created_utc": now(),
        "policy_path": args.policy,
        "status": "full_runtime_ready"
        if full_runtime_ready and policy_ready
        else "bridge_ready"
        if bridge_runtime_ready
        else "runtime_blocked",
        "summary": {
            "local_minecraft_install_detected": bool(install_paths),
            "local_install_count": len(install_paths),
            "user_license_attested": policy_ready,
            "java_available": bool(java.get("available")),
            "node_available": bool(node.get("available")),
            "full_minecraft_runtime_ready": bool(full_runtime_ready and policy_ready),
            "bridge_runtime_ready": bridge_runtime_ready,
            "external_inference_calls": 0,
        },
        "install_paths": install_paths,
        "runtime": {
            "java": java,
            "node": node,
            "python_modules": modules,
        },
        "source_cards": source_cards,
        "checks": [
            check("user_license_attested", policy_ready, "local runtime still requires launcher/user-owned account"),
            check("local_install_detected", bool(install_paths), ", ".join(row["path"] for row in install_paths) or "not found"),
            check("java_available", bool(java.get("available")), java.get("version") or java.get("error") or ""),
            check(
                "full_harness_module_available",
                bool(full_runtime_ready),
                "minedojo/malmo/minerl module plus Java/local install required",
            ),
            check("bridge_world_available", bridge_runtime_ready, "crafter or craftax import available"),
            check("no_public_server_by_default", True, "policy forbids public server autonomy by default"),
            check("no_credentials_stored", True, "probe reads no account tokens or launcher credentials"),
            check("external_inference_zero", True, "only local runtime metadata inspected"),
        ],
        "next_actions": next_actions(install_paths, java, modules, source_cards),
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def detect_install_paths(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = []
    for raw in policy.get("local_install_paths", []):
        if isinstance(raw, str):
            candidates.append(expand_env_path(raw))
    candidates.extend(
        [
            Path(os.environ.get("APPDATA", "")) / ".minecraft",
            Path(os.environ.get("USERPROFILE", "")) / "AppData" / "Roaming" / ".minecraft",
        ]
    )
    seen: set[str] = set()
    for path in candidates:
        if not path:
            continue
        resolved = path.expanduser()
        key = str(resolved).lower()
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        rows.append(
            {
                "path": rel_or_abs(resolved),
                "kind": classify_install_path(resolved),
                "launcher_profiles": (resolved / "launcher_profiles.json").exists(),
                "versions_count": len(list((resolved / "versions").glob("*"))) if (resolved / "versions").exists() else 0,
                "saves_count": len(list((resolved / "saves").glob("*"))) if (resolved / "saves").exists() else 0,
            }
        )
    return rows


def expand_env_path(raw: str) -> Path:
    text = raw
    for key, value in os.environ.items():
        text = text.replace(f"%{key}%", value)
    return Path(text)


def classify_install_path(path: Path) -> str:
    if path.name == ".minecraft":
        return "java_edition"
    if "com.mojang" in str(path).lower():
        return "bedrock_edition_local_state"
    return "minecraft_related"


def module_status() -> dict[str, dict[str, Any]]:
    modules = {
        "minerl": "minerl",
        "minedojo": "minedojo",
        "malmoenv": "malmoenv",
        "MalmoPython": "MalmoPython",
        "crafter": "crafter",
        "craftax": "craftax",
    }
    return {label: module_available(module) | {"module": module} for label, module in modules.items()}


def source_card_status() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for source_id in ["minerl", "minedojo", "malmo", "voyager_minecraft", "crafter", "craftax"]:
        card = ROOT / "benchmarks" / "cards" / f"source_{source_id}.json"
        out[source_id] = {"card_path": rel_or_abs(card), "card_exists": card.exists()}
    return out


def next_actions(
    install_paths: list[dict[str, Any]],
    java: dict[str, Any],
    modules: dict[str, dict[str, Any]],
    source_cards: dict[str, dict[str, Any]],
) -> list[str]:
    actions: list[str] = []
    if not install_paths:
        actions.append("Install or locate a licensed local Minecraft runtime before full Minecraft harness pressure.")
    if not java.get("available"):
        actions.append("Install or expose Java for Java Edition harnesses such as MineDojo, Malmo, or MineRL.")
    if not any(modules[key]["available"] for key in ["crafter", "craftax"]):
        actions.append("Use resource pantry and adapter smoke to stage Crafter/Craftax as Minecraft-like bridge pressure.")
    if not any(modules[key]["available"] for key in ["minedojo", "malmoenv", "MalmoPython", "minerl"]):
        actions.append("Stage one full Minecraft harness under the license policy when ready.")
    missing_cards = [key for key, value in source_cards.items() if not value.get("card_exists")]
    if missing_cards:
        actions.append("Refresh online source catalog and adapter factory for missing Minecraft cards: " + ", ".join(missing_cards))
    if not actions:
        actions.append("Run minecraft_rl pressure and export traces/residuals for transfer.")
    return actions[:6]


def command_version(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=10)
    except Exception as exc:  # noqa: BLE001 - diagnostics only.
        return {"available": False, "error": str(exc)}
    text = (result.stdout or result.stderr).strip().splitlines()
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "version": text[0] if text else "",
    }


def module_available(module: str) -> dict[str, Any]:
    local = importlib.util.find_spec(module) is not None
    if local:
        return {"available": True, "python": rel_or_abs(Path(sys.executable)), "source": "current_interpreter"}
    for python in candidate_pythons():
        result = subprocess.run(
            [str(python), "-c", f"import {module}; print(getattr({module}, '__version__', 'import_ok'))"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=15,
        )
        if result.returncode == 0:
            return {
                "available": True,
                "python": rel_or_abs(python),
                "source": "candidate_interpreter",
                "stdout": result.stdout.strip()[-400:],
            }
    return {"available": False, "python": "", "source": "not_found"}


def candidate_pythons() -> list[Path]:
    paths = [
        ROOT / ".venv-minecraft-rl-py311" / "Scripts" / "python.exe",
        ROOT / ".venv-puffer" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": str(evidence)[:1200]}


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
