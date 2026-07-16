from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from open_code_training_pantry import (  # noqa: E402
    iter_tar_source_files,
    language_for_extension,
    repos_from_config,
)


class OpenCodeTrainingPantryTests(unittest.TestCase):
    def test_repo_config_deduplicates_and_respects_disabled_rows(self) -> None:
        config = {
            "repos": [
                {"repo": "owner/one"},
                {"repo": "owner/one"},
                {"repo": "owner/two", "enabled": False},
                "owner/three",
            ]
        }
        self.assertEqual(["owner/one", "owner/three"], repos_from_config(config))

    def test_web_and_rust_extensions_map_to_canonical_languages(self) -> None:
        self.assertEqual("typescript", language_for_extension(".tsx"))
        self.assertEqual("javascript", language_for_extension(".jsx"))
        self.assertEqual("html", language_for_extension(".html"))
        self.assertEqual("css", language_for_extension(".css"))
        self.assertEqual("css", language_for_extension(".scss"))
        self.assertEqual("css", language_for_extension(".less"))
        self.assertEqual("rust", language_for_extension(".rs"))

    def test_source_only_inventory_is_order_independent_and_fail_closed(self) -> None:
        entries = [
            ("fixture-main/src/zeta.py", b"def zeta(value):\n    return value + 1\n"),
            ("fixture-main/src/alpha.py", b"def alpha(value):\n    return value * 2\n"),
            ("fixture-main/benchmarks/humaneval_case.py", b"def leaked():\n    return 1\n"),
            ("fixture-main/src/Duplicate.py", b"def first():\n    return 1\n"),
            ("fixture-main/src/duplicate.py", b"def second():\n    return 2\n"),
            ("fixture-main/src/oversized.py", b"x" * 4096),
            ("fixture-main/README.md", b"not an admitted source extension"),
        ]
        parent = Path(tempfile.mkdtemp())
        try:
            first = parent / "first.tar.gz"
            second = parent / "second.tar.gz"
            write_tar(first, entries)
            write_tar(second, list(reversed(entries)))

            kwargs = {
                "repo": "owner/fixture",
                "license_spdx": "MIT",
                "max_files": 10,
                "max_bytes": 1024,
            }
            rows_a = iter_tar_source_files(first, **kwargs)
            rows_b = iter_tar_source_files(second, **kwargs)

            identity_a = [(row["path"], row["sha256"]) for row in rows_a]
            identity_b = [(row["path"], row["sha256"]) for row in rows_b]
            self.assertEqual(identity_a, identity_b)
            self.assertEqual(
                ["src/alpha.py", "src/zeta.py"],
                [row["path"] for row in rows_a],
            )
            self.assertTrue(all(row["benchmark_excluded"] for row in rows_a))

            bounded = iter_tar_source_files(first, **{**kwargs, "max_files": 1})
            self.assertEqual(["src/alpha.py"], [row["path"] for row in bounded])
        finally:
            for path in parent.glob("*"):
                path.unlink()
            parent.rmdir()
        self.assertFalse(parent.exists())


def write_tar(path: Path, entries: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))


if __name__ == "__main__":
    unittest.main()
