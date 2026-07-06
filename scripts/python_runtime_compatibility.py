"""Report Python runtime availability for governed SparkStream arms.

The drone competition lane may need a Python 3.14 environment while the rest
of the project currently uses the existing local orchestration/runtime envs.
This script detects what is available and writes a machine-readable report.
It never installs Python and never mutates environments.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "python_runtime_policy.json"
DEFAULT_OUT = ROOT / "reports" / "python_runtime_compatibility.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    args = parser.parse_args()

    config = read_json(ROOT / args.config)
    discovered = discover_python_launchers()
    arms = [
        evaluate_arm(arm, discovered)
        for arm in config.get("runtime_arms", [])
        if isinstance(arm, dict)
    ]
    report = {
        "policy": "sparkstream_python_runtime_compatibility_v0",
        "created_utc": now(),
        "config": str(Path(args.config)).replace("\\", "/"),
        "current_python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "discovered_interpreters": discovered,
        "arms": arms,
        "summary": summarize(arms),
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def discover_python_launchers() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(interpreter_row(Path(sys.executable), "current"))
    venv_candidates = [
        ROOT / ".venv-puffer" / "Scripts" / "python.exe",
        ROOT / ".venv-drone-py314" / "Scripts" / "python.exe",
    ]
    for candidate in venv_candidates:
        if candidate.exists():
            rows.append(interpreter_row(candidate, "venv"))
    try:
        proc = subprocess.run(["py", "-0p"], cwd=ROOT, text=True, capture_output=True, timeout=10)
    except Exception:
        proc = None
    if proc and proc.returncode == 0:
        for line in proc.stdout.splitlines():
            match = re.search(r"-(\d+\.\d+)[^\s]*\s+(.+python(?:\.exe)?)", line, re.IGNORECASE)
            if not match:
                continue
            executable = Path(match.group(2).strip())
            if not executable.is_absolute():
                continue
            rows.append(interpreter_row(executable, f"py_launcher_{match.group(1)}"))
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("executable") or "").lower()
        if key:
            dedup[key] = row
    return list(dedup.values())


def interpreter_row(path: Path, source: str) -> dict[str, Any]:
    exists = path.exists()
    version = ""
    if exists:
        try:
            proc = subprocess.run(
                [str(path), "--version"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=10,
            )
            version = (proc.stdout or proc.stderr).strip().replace("Python ", "")
        except Exception as exc:  # noqa: BLE001 - diagnostic report path.
            version = f"error:{exc}"
    return {
        "source": source,
        "executable": str(path),
        "exists": exists,
        "version": version,
        "prefix": ".".join(version.split(".")[:2]) if version and not version.startswith("error:") else "",
    }


def evaluate_arm(arm: dict[str, Any], discovered: list[dict[str, Any]]) -> dict[str, Any]:
    required = str(arm.get("required_python_prefix") or "")
    preferred_venv = str(arm.get("preferred_venv") or "")
    preferred_path = ROOT / preferred_venv / "Scripts" / "python.exe" if preferred_venv else None
    preferred = interpreter_row(preferred_path, "preferred_venv") if preferred_path else {}
    matching = [row for row in discovered if str(row.get("prefix")) == required]
    preferred_ok = bool(preferred.get("exists") and str(preferred.get("prefix")) == required)
    available = preferred_ok or bool(matching)
    status = "available" if available else "missing_required_python"
    return {
        "id": arm.get("id"),
        "role": arm.get("role"),
        "required_python_prefix": required,
        "known_good_python": arm.get("known_good_python", ""),
        "preferred_venv": preferred_venv,
        "preferred_interpreter": preferred,
        "matching_interpreters": matching,
        "status": status,
        "fallback_allowed": bool(arm.get("fallback_allowed")),
        "fallback_note": arm.get("fallback_note", ""),
        "risk": arm.get("risk"),
        "safety": arm.get("safety", {}),
        "next_action": next_action(arm, status),
    }


def next_action(arm: dict[str, Any], status: str) -> str:
    if status == "available":
        return "Use the preferred isolated runtime for this arm."
    if arm.get("id") == "ai_grand_prix_drone":
        return "Install Python 3.14.x on Windows and create .venv-drone-py314 before official competition runs."
    return "Create the preferred venv or update the runtime policy."


def summarize(arms: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [arm for arm in arms if arm.get("status") != "available"]
    high_risk_missing = [arm for arm in missing if arm.get("risk") in {"high", "critical"}]
    return {
        "arm_count": len(arms),
        "available": len(arms) - len(missing),
        "missing": len(missing),
        "high_risk_missing": len(high_risk_missing),
        "ai_grand_prix_runtime_ready": any(
            arm.get("id") == "ai_grand_prix_drone" and arm.get("status") == "available"
            for arm in arms
        ),
        "status": "GREEN" if not missing else ("YELLOW" if not high_risk_missing else "YELLOW_RUNTIME_ACTION_REQUIRED"),
    }


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
