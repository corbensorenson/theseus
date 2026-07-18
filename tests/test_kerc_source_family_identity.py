from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from kerc_source_family_identity import (  # noqa: E402
    family_identity_receipts,
    source_family,
)


def identities(path: Path) -> dict[str, str]:
    receipts = family_identity_receipts(
        source_path=path,
        source_label="fixture.py",
        role="fixture",
        family_roots={"one": "family_one", "two": "family_two"},
        external_paths={},
    )
    return {family: row["identity_sha256"] for family, row in receipts.items()}


def test_family_identity_invalidates_only_transitive_dependants(tmp_path: Path) -> None:
    path = tmp_path / "fixture.py"
    path.write_text(
        "COMMON = 3\n"
        "ONE = 5\n"
        "TWO = 7\n"
        "def shared(value):\n    return value + COMMON\n"
        "def family_one(value):\n    return shared(value) + ONE\n"
        "def family_two(value):\n    return shared(value) + TWO\n",
        encoding="utf-8",
    )
    original = identities(path)

    path.write_text(path.read_text(encoding="utf-8").replace("ONE = 5", "ONE = 11"))
    family_local = identities(path)
    assert family_local["one"] != original["one"]
    assert family_local["two"] == original["two"]

    path.write_text(
        path.read_text(encoding="utf-8").replace("COMMON = 3", "COMMON = 13")
    )
    common_change = identities(path)
    assert common_change["one"] != family_local["one"]
    assert common_change["two"] != family_local["two"]


def test_unrelated_source_edit_does_not_invalidate_family_identity(tmp_path: Path) -> None:
    path = tmp_path / "fixture.py"
    path.write_text(
        "def family_one(value):\n    return value + 1\n"
        "def family_two(value):\n    return value + 2\n",
        encoding="utf-8",
    )
    original = identities(path)
    path.write_text(
        path.read_text(encoding="utf-8")
        + "def unrelated(value):\n    return value * 1000\n",
        encoding="utf-8",
    )
    assert identities(path) == original


def test_family_specific_external_dependency_has_selective_invalidation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fixture.py"
    common = tmp_path / "common.py"
    special = tmp_path / "special.py"
    source.write_text(
        "def family_one(value):\n    return value + 1\n"
        "def family_two(value):\n    return value + 2\n",
        encoding="utf-8",
    )
    common.write_text("COMMON = 1\n", encoding="utf-8")
    special.write_text("SPECIAL = 1\n", encoding="utf-8")

    def receipts() -> dict[str, str]:
        rows = family_identity_receipts(
            source_path=source,
            source_label="fixture.py",
            role="fixture",
            family_roots={"one": "family_one", "two": "family_two"},
            external_paths={"common": common},
            family_external_paths={"one": {"special": special}},
        )
        return {family: row["identity_sha256"] for family, row in rows.items()}

    original = receipts()
    special.write_text("SPECIAL = 2\n", encoding="utf-8")
    family_local = receipts()
    assert family_local["one"] != original["one"]
    assert family_local["two"] == original["two"]

    common.write_text("COMMON = 2\n", encoding="utf-8")
    common_change = receipts()
    assert common_change["one"] != family_local["one"]
    assert common_change["two"] != family_local["two"]


def test_source_family_uses_only_dataset_and_source_identity() -> None:
    assert source_family(dataset_key="dolly", source_id="dolly:1") == "dolly_direct"
    assert (
        source_family(dataset_key="dolly", source_id="dolly-grounded:1")
        == "dolly_grounded"
    )
    assert (
        source_family(dataset_key="masc", source_id="masc-event-coref:1")
        == "masc_event_coreference"
    )
    assert (
        source_family(dataset_key="masc", source_id="masc-mpqa-relation:1")
        == "masc_mpqa_relation"
    )
    assert (
        source_family(dataset_key="gum", source_id="gum-erst:GUM_academic_art:2")
        == "gum_discourse"
    )
    assert (
        source_family(dataset_key="oasst2", source_id="oasst2-behavior:1")
        == "oasst_behavior"
    )
