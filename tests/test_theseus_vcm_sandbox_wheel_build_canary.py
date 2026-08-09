from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_sandbox_wheel_build_canary as owner  # noqa:E402
CONFIG=ROOT/"configs/theseus_vcm_sandbox_wheel_build_canary.json"
def test_preflight_binds_exact_prequalified_sdist()->None:
 cfg,bound,faults=owner.preflight(CONFIG);assert faults==[];assert owner.p2a.sha256_file(bound["sdist"])==cfg["sdist_sha256"];assert set(bound["tools"])=={"uv","python","sandbox_exec"}
def test_canary_authority_is_build_only()->None:
 cfg=owner.p2a.read_json(CONFIG);assert cfg["authority"]["single_exact_sdist_build_authorized"] is True;assert cfg["authority"]["network_denied_sandbox_build_authorized"] is True;assert cfg["authority"]["parent_target_evaluator_execution_authorized"] is False;assert cfg["authority"]["local_model_calls_authorized"] is False;assert cfg["authority"]["external_reference_calls_authorized"] is False
