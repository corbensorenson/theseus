from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_task36_dependency_canary as owner  # noqa:E402
def test_preflight_seals_all_32_checksums_without_build_authority()->None:
 path=ROOT/"configs"/"theseus_vcm_task36_dependency_canary.json";r=owner.preflight(json.loads(path.read_text()),path)
 assert r["trigger_state"]=="PAUSED" and r["lock_checksum_package_count"]==32 and r["lock_checksum_identity_sha256"]=="d5f9f546d3892877a83ca1b16d235b8e5f1860bcc937dfade65b0897b8c4eacc"
 assert r["dependency_installations"]==r["repository_build_executions"]==r["repository_runner_executions"]==r["candidate_or_control_calls"]==r["external_reference_calls"]==0
def test_offline_command_adds_only_offline()->None:
 c=json.loads((ROOT/"configs"/"theseus_vcm_task36_dependency_canary.json").read_text());assert c["commands"]["offline_replay_args"]==[*c["commands"]["online_fetch_args"],"--offline"] and c["authority"]["cargo_build_authorized"] is False
