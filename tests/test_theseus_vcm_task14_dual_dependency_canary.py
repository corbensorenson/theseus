from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_task14_dual_dependency_canary as owner  # noqa:E402
def test_preflight_seals_two_distinct_build_free_closures()->None:
 p=ROOT/"configs"/"theseus_vcm_task14_dual_dependency_canary.json";r=owner.preflight(json.loads(p.read_text()),p)
 assert r["trigger_state"]=="PAUSED" and set(r["observed_side_dependency_identities"])=={"parent","target"}
 assert r["observed_side_dependency_identities"]["parent"]!=r["observed_side_dependency_identities"]["target"]
 assert r["source_build_executions"]==r["project_installations"]==r["repository_runner_executions"]==r["candidate_or_control_calls"]==r["external_reference_calls"]==0
def test_commands_disable_builds_projects_downloads_and_add_only_offline()->None:
 c=json.loads((ROOT/"configs"/"theseus_vcm_task14_dual_dependency_canary.json").read_text());online=c["commands"]["online_sync_args"]
 for flag in ("--frozen","--no-build","--no-install-project","--no-install-workspace","--no-python-downloads"):assert flag in online
 assert c["commands"]["offline_replay_args"]==[*online,"--offline"]
