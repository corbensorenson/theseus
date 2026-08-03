from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_task30_dependency_canary as owner  # noqa:E402
def test_preflight_binds_single_identical_fifty_checksum_closure()->None:
 p=ROOT/"configs"/"theseus_vcm_task30_dependency_canary.json";r=owner.preflight(json.loads(p.read_text()),p)
 assert r["trigger_state"]=="PAUSED" and r["lock_checksum_package_count"]==50
 assert r["observed_parent_target_dependency_identities"]["parent"]==r["observed_parent_target_dependency_identities"]["target"]
 for k in ("dependency_installations","repository_build_executions","repository_runner_executions","parent_target_or_evaluator_executions","candidate_or_control_calls","external_reference_calls"):assert r[k]==0
def test_commands_are_locked_fetch_only_and_offline_delta()->None:
 c=json.loads((ROOT/"configs"/"theseus_vcm_task30_dependency_canary.json").read_text());online=c["commands"]["online_fetch_args"]
 assert online==["fetch","--locked","--config","net.git-fetch-with-cli=false"] and c["commands"]["offline_replay_args"]==[*online,"--offline"]
