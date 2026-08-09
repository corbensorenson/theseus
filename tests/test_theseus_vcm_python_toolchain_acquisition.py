import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_python_toolchain_acquisition as owner
CONFIG=ROOT/"configs"/"theseus_vcm_python313_toolchain_acquisition.json"
def test_preflight_authorizes_only_exact_toolchain_acquisition():
 r=owner.preflight_report(CONFIG);assert r["trigger_state"]=="PAUSED";assert r["execution_performed"] is False;assert r["package_installations"]==0;assert r["repository_executions"]==0;assert r["candidate_or_control_calls"]==0;assert r["project_selected_output_cap"] is None
def test_green_receipt_binds_exact_python_when_present():
 cfg=json.loads(CONFIG.read_text());path=ROOT/cfg["report"]
 if not path.is_file():return
 r=json.loads(path.read_text())
 if r.get("trigger_state")!="GREEN":return
 assert r["interpreter"]["version"]==cfg["version"];assert r["version_probe_executions"]==1;assert r["global_bin_registration_performed"] is False;assert r["package_installations"]==0
