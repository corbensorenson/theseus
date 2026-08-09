from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_sdist_batch_preflight as owner  # noqa:E402
CONFIG=ROOT/"configs/theseus_vcm_sdist_batch_preflight.json"
def test_preflight_binds_exact_three_row_batch()->None:
 cfg,_,faults=owner.preflight(CONFIG);assert faults==[];assert [(r["name"],r["version"]) for r in cfg["packages"]]==[("habluetooth","6.26.5"),("bluetooth-data-tools","1.29.21"),("PyRIC","0.1.6.3")]
def test_batch_has_no_build_or_evaluator_authority()->None:
 cfg=owner.p2a.read_json(CONFIG);assert cfg["authority"]["source_build_authorized"] is False;assert cfg["authority"]["package_installation_authorized"] is False;assert cfg["authority"]["parent_target_evaluator_execution_authorized"] is False;assert cfg["authority"]["local_model_calls_authorized"] is False;assert cfg["authority"]["external_reference_calls_authorized"] is False
