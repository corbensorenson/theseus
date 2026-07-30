from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import training_data_admission_v1 as admission  # noqa: E402


def decision(value: str) -> str:
    return admission.license_decision(
        Counter({admission.normalize_license(value): 1}),
        private_train_path=False,
        dogfood_path=False,
    )


def test_project_generated_license_is_admitted_canonically() -> None:
    assert decision("project_generated").startswith(
        "allowed_"
    ), "request_contract:project_generated_license_admitted"
    assert decision("project-generated").startswith(
        "allowed_"
    ), "request_contract:project_generated_hyphen_license_admitted"


def test_private_generated_license_is_admitted_canonically() -> None:
    assert decision("private_generated").startswith(
        "allowed_"
    ), "request_contract:private_generated_license_admitted"
    assert decision("private-generated").startswith(
        "allowed_"
    ), "request_contract:private_generated_hyphen_license_admitted"


def test_near_miss_generated_license_remains_blocked() -> None:
    assert decision("project_generated_extra").startswith("blocked_license_")
