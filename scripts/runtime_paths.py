"""CLI for Project Theseus runtime path setup and generated-directory migration."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import theseus_runtime


ROOT = theseus_runtime.ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status")
    status.add_argument("--create", action="store_true")
    status.add_argument("--out", default="reports/runtime_paths.json")

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--out", default="reports/macos_runtime_doctor.json")

    init = sub.add_parser("init")
    init.add_argument("--runtime-root", default="")
    init.add_argument("--no-create", action="store_true")
    init.add_argument("--out", default="reports/runtime_paths.json")

    migrate = sub.add_parser("migrate-junctions")
    migrate.add_argument("--runtime-root", default="")
    migrate.add_argument("--dirs", default="reports,checkpoints,target")
    migrate.add_argument("--dry-run", action="store_true")
    migrate.add_argument("--out", default="reports/runtime_paths_migration.json")

    args = parser.parse_args()
    if args.command in {None, "status"}:
        report = theseus_runtime.runtime_report(create=bool(getattr(args, "create", False)), write_report=True)
    elif args.command == "doctor":
        report = theseus_runtime.runtime_doctor_report(write_report=True)
    elif args.command == "init":
        report = theseus_runtime.write_local_config(args.runtime_root, create=not args.no_create)
    elif args.command == "migrate-junctions":
        if args.runtime_root:
            theseus_runtime.write_local_config(args.runtime_root, create=True)
        report = migrate_junctions(args.dirs, dry_run=bool(args.dry_run))
    else:
        parser.print_help()
        return 2
    out = ROOT / getattr(args, "out", "reports/runtime_paths.json")
    theseus_runtime.write_json(out, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) else 2


def migrate_junctions(dirs_csv: str, *, dry_run: bool) -> dict[str, Any]:
    runtime = theseus_runtime.runtime_report(create=True, write_report=True)
    paths = runtime["paths"]
    runtime_root = Path(paths["runtime_root"]["path"])
    mapping = {
        "reports": Path(paths["reports_dir"]["path"]),
        "checkpoints": Path(paths["checkpoints_dir"]["path"]),
        "target": Path(paths["cargo_target_dir"]["path"]),
        "dist": runtime_root / "workspace-dist",
        "games": runtime_root / "workspace-games",
        "data": runtime_root / "workspace-data",
        "vendor": runtime_root / "workspace-vendor",
        ".venv-puffer": runtime_root / "venvs" / ".venv-puffer",
        ".venv-emulator-py311": runtime_root / "venvs" / ".venv-emulator-py311",
        ".venv-minecraft-rl-py311": runtime_root / "venvs" / ".venv-minecraft-rl-py311",
        ".venv-drone-pyflyt-py311": runtime_root / "venvs" / ".venv-drone-pyflyt-py311",
        ".venv-drone-gym-pybullet-py311": runtime_root / "venvs" / ".venv-drone-gym-pybullet-py311",
        ".venv-drone-control-py311": runtime_root / "venvs" / ".venv-drone-control-py311",
        ".venv-drone-racing-py311": runtime_root / "venvs" / ".venv-drone-racing-py311",
    }
    requested = [item.strip() for item in dirs_csv.split(",") if item.strip()]
    actions = []
    for name in requested:
        if name not in mapping:
            actions.append({"name": name, "ok": False, "error": "unsupported_directory"})
            continue
        source = (ROOT / name).resolve()
        target = mapping[name].resolve()
        actions.append(migrate_one(name, source, target, dry_run=dry_run))
    return {
        "ok": all(row.get("ok") for row in actions),
        "policy": "project_theseus_runtime_migration_v0",
        "dry_run": dry_run,
        "actions": actions,
        "runtime": runtime,
    }


def migrate_one(name: str, source: Path, target: Path, *, dry_run: bool) -> dict[str, Any]:
    workspace = ROOT.resolve()
    runtime_root = Path(theseus_runtime.runtime_report()["paths"]["runtime_root"]["path"]).resolve()
    if not str(source).lower().startswith(str(workspace).lower()):
        return {"name": name, "ok": False, "error": "source_outside_workspace", "source": str(source)}
    if not str(target).lower().startswith(str(runtime_root).lower()):
        return {"name": name, "ok": False, "error": "target_outside_runtime_root", "target": str(target)}
    if not source.exists():
        return {"name": name, "ok": True, "status": "source_missing", "source": str(source), "target": str(target)}
    if theseus_runtime.is_reparse_point(source):
        return {"name": name, "ok": True, "status": "already_redirected", "source": str(source), "target": str(target)}
    source_size_gib = theseus_runtime.directory_size_gib(source)
    target_had_files = target.exists() and any(target.iterdir())
    if dry_run:
        return {
            "name": name,
            "ok": True,
            "status": "would_resume_copy_and_link" if target_had_files else "would_move_and_link",
            "source": str(source),
            "target": str(target),
            "size_gib": source_size_gib,
            "target_had_files": target_had_files,
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.is_file():
        return {"name": name, "ok": False, "error": "target_exists_file", "source": str(source), "target": str(target)}
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        remove_source_report = remove_path_with_retries(source)
    else:
        # Cross-drive shutil.move copies first and then removes the source tree. If
        # cleanup hits a live Windows handle, a later run must be able to resume.
        shutil.copytree(source, target, dirs_exist_ok=True)
        remove_source_report = remove_tree_with_retries(source)
    if not remove_source_report["ok"]:
        return {
            "name": name,
            "ok": False,
            "status": "copied_but_source_locked",
            "source": str(source),
            "target": str(target),
            "source_size_gib": source_size_gib,
            "target_had_files": target_had_files,
            "remove": remove_source_report,
        }
    create_directory_link(target, source)
    return {
        "name": name,
        "ok": True,
        "status": "resumed_copy_and_linked" if target_had_files else "moved_and_linked",
        "source": str(source),
        "target": str(target),
        "source_size_gib": source_size_gib,
        "target_had_files": target_had_files,
    }


def remove_path_with_retries(path: Path, *, attempts: int = 5, delay_seconds: float = 0.5) -> dict[str, Any]:
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            path.unlink()
            return {"ok": True, "attempt": attempt}
        except FileNotFoundError:
            return {"ok": True, "attempt": attempt, "status": "already_removed"}
        except OSError as exc:
            errors.append(f"attempt {attempt}: {exc}")
            time.sleep(delay_seconds)
    return {"ok": False, "errors": errors}


def remove_tree_with_retries(path: Path, *, attempts: int = 5, delay_seconds: float = 0.5) -> dict[str, Any]:
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            shutil.rmtree(path)
            return {"ok": True, "attempt": attempt}
        except FileNotFoundError:
            return {"ok": True, "attempt": attempt, "status": "already_removed"}
        except OSError as exc:
            errors.append(f"attempt {attempt}: {exc}")
            time.sleep(delay_seconds)
    return {"ok": False, "errors": errors}


def create_directory_link(target: Path, link: Path) -> None:
    if platform.system() == "Windows":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
    else:
        os.symlink(target, link, target_is_directory=True)


if __name__ == "__main__":
    raise SystemExit(main())
