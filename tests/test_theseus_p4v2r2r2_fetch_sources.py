from __future__ import annotations

import gzip
import io
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2r2_fetch_sources as fetch  # noqa: E402


def write_archive(path: Path, members: dict[str, bytes]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as handle:
                for name, payload in members.items():
                    info = tarfile.TarInfo(name=name)
                    info.size = len(payload)
                    handle.addfile(info, io.BytesIO(payload))


def test_sealed_registry_binds_transport_before_fetch() -> None:
    report = fetch.source_registry.audit(fetch.REGISTRY)

    assert report["trigger_state"] == "GREEN"
    assert fetch.sha256_file(fetch.REGISTRY) == fetch.EXPECTED_REGISTRY_SHA256
    assert report["candidate_or_control_calls"] == 0
    assert report["archive_fetches"] == 0


def test_projection_is_deterministic_and_omits_unrequired_members(tmp_path: Path) -> None:
    source = tmp_path / "source.tar.gz"
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    write_archive(
        source,
        {
            "repo-sha/LICENSE": b"MIT\n",
            "repo-sha/src/module.py": b"value = 1\n",
            "repo-sha/tests/hidden.py": b"secret = 2\n",
        },
    )

    one = fetch.project_archive(
        source, first, root="repo-sha", relative_paths=["LICENSE", "src/module.py"]
    )
    two = fetch.project_archive(
        source, second, root="repo-sha", relative_paths=["src/module.py", "LICENSE"]
    )

    assert one["output_sha256"] == two["output_sha256"]
    with tarfile.open(first, "r:gz") as handle:
        names = [member.name for member in handle.getmembers()]
    assert names == ["repo-sha/LICENSE", "repo-sha/src/module.py"]
    assert "repo-sha/tests/hidden.py" not in names
