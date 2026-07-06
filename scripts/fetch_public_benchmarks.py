"""Fetch public benchmark definitions for local SymLiquid evaluation.

This script downloads benchmark code/data definitions only. It does not call
hosted model APIs, run external inference, or submit anything to a service.
Other systems' scores should remain public metadata; SymLiquid runs locally.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path


REPOS = {
    "blimp": {
        "url": "https://github.com/alexwarstadt/blimp.git",
        "purpose": "BLIMP grammatical acceptability benchmark definitions.",
    },
    "babylm_eval": {
        "url": "https://github.com/babylm/evaluation-pipeline.git",
        "purpose": "BabyLM-style evaluation harness references.",
    },
    "pufferlib": {
        "url": "https://github.com/PufferAI/PufferLib.git",
        "purpose": "PufferLib/Ocean environment references for local rollout adapters.",
    },
}


def run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def fetch_repo(name: str, spec: dict[str, str], root: Path, update: bool) -> dict[str, str]:
    target = root / name
    url = spec["url"]
    record = {
        "name": name,
        "url": url,
        "purpose": spec["purpose"],
        "path": str(target),
        "status": "unknown",
        "commit": "",
        "error": "",
    }
    try:
        if target.exists():
            if update:
                run(["git", "fetch", "--depth", "1", "origin"], cwd=target)
                run(["git", "reset", "--hard", "FETCH_HEAD"], cwd=target)
                record["status"] = "updated"
            else:
                record["status"] = "existing"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", "--depth", "1", url, str(target)])
            record["status"] = "cloned"
        record["commit"] = run(["git", "rev-parse", "HEAD"], cwd=target)
    except subprocess.CalledProcessError as exc:
        record["status"] = "failed"
        record["error"] = (exc.stderr or exc.stdout or str(exc)).strip()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="data/public_benchmarks")
    parser.add_argument(
        "--repos",
        default="blimp,babylm_eval,pufferlib",
        help="Comma-separated repo keys to fetch.",
    )
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_root).resolve()
    requested = [name.strip() for name in args.repos.split(",") if name.strip()]
    unknown = [name for name in requested if name not in REPOS]
    if unknown:
        raise SystemExit(f"Unknown repo keys: {', '.join(unknown)}")

    records = [fetch_repo(name, REPOS[name], root, args.update) for name in requested]
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy": {
            "external_inference": "forbidden",
            "public_scores": "metadata_only",
            "notes": "Fetched repos are benchmark definitions or local environment code, not hosted model endpoints.",
        },
        "repos": records,
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
