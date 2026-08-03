from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_rust_toolchain_bootstrap as owner  # noqa: E402
def test_preflight_seals_exact_minimal_rust_without_cargo_or_repository_authority()->None:
    path=ROOT/"configs"/"theseus_vcm_rust_toolchain_bootstrap.json"; report=owner.preflight(json.loads(path.read_text()),path)
    assert report["trigger_state"]=="PAUSED" and report["toolchain_acquisition_executions"]==0
    assert report["cargo_dependency_fetches"]==report["repository_executions"]==report["candidate_or_control_calls"]==report["external_reference_calls"]==0
