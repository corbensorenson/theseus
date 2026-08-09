from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_sdist_batch_build as owner  # noqa:E402
CONFIG=ROOT/"configs/theseus_vcm_sdist_batch_build.json"
def test_preflight_binds_three_pinned_build_profiles()->None:
 cfg,bound,faults=owner.preflight(CONFIG);assert faults==[];assert len(bound["rows"])==3;assert all(all("==" in value for value in row["pinned_build_requirements"]) for row in cfg["rows"])
def test_build_batch_has_no_evaluator_or_model_authority()->None:
 cfg=owner.p2a.read_json(CONFIG);assert cfg["authority"]["network_denied_sandbox_build_authorized"] is True;assert cfg["authority"]["parent_target_evaluator_execution_authorized"] is False;assert cfg["authority"]["local_model_calls_authorized"] is False;assert cfg["authority"]["external_reference_calls_authorized"] is False
