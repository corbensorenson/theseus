from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_untrusted_build_preflight as owner  # noqa: E402

CONFIG = ROOT / "configs" / "theseus_vcm_untrusted_build_preflight.json"


def test_preflight_binds_exact_task13_sdist_wall() -> None:
    cfg, bound, faults = owner.preflight(CONFIG)
    assert faults == []
    assert cfg["package"]["name"] == "mock-open"
    assert cfg["package"]["version"] == "1.4.0"
    assert cfg["package"]["sha256"] == "c3ecb6b8c32a5899a4f5bf4495083b598b520c698bba00e1ce2ace6e9c239100"
    assert bound["task13"]["disposition"] == "INCONCLUSIVE_EXPERIMENT_DEPENDENCY_RESOLUTION"
    assert cfg["curl"]["path"] == "/usr/bin/curl"


def test_preflight_has_no_build_or_evaluator_authority() -> None:
    cfg = owner.p2a.read_json(CONFIG)
    assert cfg["authority"]["source_build_authorized"] is False
    assert cfg["authority"]["build_backend_execution_authorized"] is False
    assert cfg["authority"]["package_installation_authorized"] is False
    assert cfg["authority"]["parent_target_evaluator_execution_authorized"] is False
    assert cfg["authority"]["local_model_calls_authorized"] is False
    assert cfg["authority"]["external_reference_calls_authorized"] is False
