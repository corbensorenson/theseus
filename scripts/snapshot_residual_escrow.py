"""Snapshot residual escrow before a candidate/frontier run."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="reports/residual_escrow.json")
    parser.add_argument("--out", default="reports/residual_escrow_pre_candidate_baseline.json")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    if not source.exists():
        raise SystemExit(f"residual escrow source not found: {source}")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, out)
    payload = read_json(out)
    marker = {
        "policy": "local_only_no_external_inference",
        "methodology": "residual_escrow_snapshot",
        "source": str(source),
        "out": str(out),
        "snapshot_unix_time": time.time(),
        "summary": payload.get("summary", {}),
        "cluster_count": len(payload.get("clusters") or []),
    }
    marker_path = out.with_suffix(".snapshot.json")
    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(marker, indent=2))
    return 0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
