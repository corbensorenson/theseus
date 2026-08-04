from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_task26_dependency_canary as owner  # noqa:E402
def test_preflight_binds_single_identical_build_free_uv_closure()->None:
 p=ROOT/"configs"/"theseus_vcm_task26_dependency_canary.json";r=owner.preflight(json.loads(p.read_text()),p)
 assert r["trigger_state"]=="PAUSED" and r["observed_parent_target_dependency_identities"]["parent"]==r["observed_parent_target_dependency_identities"]["target"]
 assert r["observed_parent_target_dependency_identities"]["target"]["package_count"]==58
 for k in ("source_build_executions","project_installations","repository_runner_executions","candidate_or_control_calls","external_reference_calls"):assert r[k]==0
def test_commands_disable_builds_projects_groups_and_downloads()->None:
 c=json.loads((ROOT/"configs"/"theseus_vcm_task26_dependency_canary.json").read_text());online=c["commands"]["online_sync_args"]
 for flag in ("--frozen","--no-build","--no-install-project","--no-install-workspace","--no-default-groups","--no-python-downloads"):assert flag in online
 assert c["commands"]["offline_replay_args"]==[*online,"--offline"]
