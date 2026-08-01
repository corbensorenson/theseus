from __future__ import annotations

import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p2b_sanitize_archive as sanitizer  # noqa: E402


def test_sanitizer_is_deterministic_and_omits_links(tmp_path: Path) -> None:
    source = tmp_path / "source"
    root = source / "repo-root"
    root.mkdir(parents=True)
    (root / "value.txt").write_text("value\n", encoding="utf-8")
    (root / "linked.txt").symlink_to("value.txt")
    archive = tmp_path / "upstream.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(root, arcname="repo-root")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    report = sanitizer.sanitize(archive, first)
    sanitizer.sanitize(archive, second)

    assert report["trigger_state"] == "GREEN"
    assert report["source_archive_root"] == "repo-root"
    assert report["omitted_members"] == [
        {"path": "repo-root/linked.txt", "kind": "symbolic_link", "linkname": "value.txt"}
    ]
    assert sanitizer.sha256_file(first) == sanitizer.sha256_file(second)
    with tarfile.open(first) as handle:
        assert "repo-root/value.txt" in handle.getnames()
        assert "repo-root/linked.txt" not in handle.getnames()
