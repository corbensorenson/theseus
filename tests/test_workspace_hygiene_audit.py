from __future__ import annotations

import scripts.theseus_workspace_hygiene_audit as hygiene


def duplicate_family(*, classified: bool) -> dict[str, object]:
    return {
        "root": "scripts",
        "family": "example",
        "count": 3,
        "classified": classified,
        "classification": (
            {
                "classification": "compatibility_wrappers",
                "canonical_surface": "example_surface",
                "canonical_path": "scripts/example.py",
                "promotion_role": "support_only",
                "successor_policy": "extend the canonical owner",
            }
            if classified
            else {}
        ),
    }


def test_classified_duplicate_families_are_not_cleanup_candidates() -> None:
    registry = {"duplicate_families": [duplicate_family(classified=True)]}

    candidates = hygiene.project_registry_candidates(registry)

    assert candidates == []


def test_unclassified_duplicate_families_remain_actionable() -> None:
    registry = {
        "duplicate_families": [
            duplicate_family(classified=True),
            duplicate_family(classified=False),
        ]
    }

    candidates = hygiene.project_registry_candidates(registry)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["kind"] == "duplicate_family_consolidation"
    assert candidate["evidence"]["count"] == 1
    assert candidate["evidence"]["classified_family_count"] == 1
