#!/usr/bin/env python3
"""Acquire one exact project-local Python toolchain for a sealed VCM instrument."""
from __future__ import annotations
import argparse,json,shutil,sys,tempfile
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_immutable_resolution_segment as prior  # noqa:E402
POLICY="project_theseus_vcm_python_toolchain_acquisition_v1";DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_python313_toolchain_acquisition.json"

def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));p.add_argument("--out",default="");p.add_argument("--execute",action="store_true");a=p.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=execute(path) if a.execute else preflight_report(path);p2a.write_json(p2a.resolve(a.out or cfg["report"]),r);print(json.dumps({k:r.get(k) for k in ("trigger_state","state","execution_performed","network_downloads","version_probe_executions","package_installations","repository_executions","candidate_or_control_calls","external_reference_calls","faults")},indent=2,sort_keys=True));return 0 if r["trigger_state"] in {"GREEN","PAUSED"} else 2

def preflight(path:Path=DEFAULT_CONFIG)->tuple[dict[str,Any],dict[str,Path],list[str]]:
 cfg=p2a.read_json(path);faults=[]
 if cfg.get("policy")!=POLICY:faults.append("policy_invalid")
 for row in p2a.dicts(cfg.get("bindings")):
  q=p2a.resolve(str(row.get("path") or ""))
  if not q.is_file() or p2a.sha256_file(q)!=row.get("sha256"):faults.append(f"binding_invalid:{row.get('id')}")
 installer=p2a.resolve(str(p2a.mapping(cfg.get("installer")).get("path") or ""));install_root=p2a.resolve(str(cfg.get("install_root") or ""));cache=p2a.resolve(str(cfg.get("cache_root") or ""));target=p2a.resolve(str(cfg.get("interpreter_path") or ""))
 if not installer.is_file() or p2a.sha256_file(installer)!=p2a.mapping(cfg["installer"]).get("sha256"):faults.append("installer_invalid")
 if install_root!=(ROOT/"runtime/vcm_evaluator/toolchains/uv-python").resolve():faults.append("install_root_invalid")
 if cache!=(ROOT/"runtime/vcm_evaluator/dependency_store/python313-acquisition-v1").resolve():faults.append("cache_root_invalid")
 expected=(install_root/str(cfg.get("toolchain_key") or "")/"bin"/"python3.13").resolve()
 if target!=expected:faults.append("interpreter_path_invalid")
 authority=p2a.mapping(cfg.get("authority"));allowed={"exact_official_toolchain_acquisition_authorized","exact_version_probe_authorized","isolated_toolchain_retention_authorized"}
 if any(authority.get(k) is not True for k in allowed) or any(v is not False for k,v in authority.items() if k not in allowed):faults.append("authority_invalid")
 return cfg,{"installer":installer,"install_root":install_root,"cache":cache,"target":target},sorted(set(faults))

def preflight_report(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg,bound,faults=preflight(path);return finish(cfg,path,bound,faults,False,{}, {},shutil.disk_usage(ROOT).free)

def execute(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg,bound,faults=preflight(path);limits=p2a.mapping(cfg.get("limits"));free_before=shutil.disk_usage(ROOT).free;reserve=int(limits["minimum_free_bytes_after_execution"]);maximum=int(limits["maximum_acquisition_and_cache_bytes"])
 if faults:return finish(cfg,path,bound,faults,False,{}, {},free_before)
 if bound["target"].exists():return finish(cfg,path,bound,["immutable_interpreter_target_already_exists"],False,{}, {},free_before)
 if free_before-maximum<reserve:return finish(cfg,path,bound,["free_space_reserve_preflight_boundary_hit"],False,{}, {},free_before)
 bound["install_root"].mkdir(parents=True,exist_ok=True);bound["cache"].mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix="theseus-vcm-python-acquisition-",dir="/private/tmp") as raw:
  home=Path(raw)/"home";temp=Path(raw)/"tmp";home.mkdir();temp.mkdir();env={"HOME":str(home),"TMPDIR":str(temp),"PATH":"/usr/bin:/bin","CI":"1","NO_COLOR":"1","LANG":"C","LC_ALL":"C","UV_NO_CONFIG":"1"};command=[str(bound["installer"]),"python","install",str(cfg["version"]),"--install-dir",str(bound["install_root"]),"--cache-dir",str(bound["cache"]),"--no-bin","--no-config","--no-progress","--color","never"]
  try:receipt=prior.run(command,ROOT,env,limits)
  except OSError as exc:receipt={"returncode":None,"duration_ms":0.0,"boundary_hit":False,"stdout":"","stderr":"","stdout_complete":True,"stderr_complete":True,"project_selected_output_cap":None,"spawn_error":{"type":type(exc).__name__,"message":str(exc)}}
  receipt["command"]=command
  probe={}
  if receipt.get("returncode")==0 and not receipt.get("boundary_hit") and bound["target"].is_file():
   probe_command=[str(bound["target"]),"--version"];probe=prior.run(probe_command,ROOT,env,limits);probe["command"]=probe_command
 observed=prior.tree_identity(bound["target"].parents[1]) if bound["target"].is_file() else {};cache=prior.tree_identity(bound["cache"]);combined=int(observed.get("bytes") or 0)+int(cache.get("bytes") or 0)
 if receipt.get("boundary_hit"):faults.append("acquisition_host_boundary_hit")
 if receipt.get("spawn_error"):faults.append("acquisition_spawn_failed")
 if receipt.get("returncode")!=0:faults.append("acquisition_returncode_nonzero")
 if not bound["target"].is_file():faults.append("interpreter_missing")
 if probe.get("returncode")!=0 or (probe.get("stdout") or probe.get("stderr") or "").strip()!=f"Python {cfg['version']}":faults.append("version_probe_invalid")
 if combined>maximum or shutil.disk_usage(ROOT).free<reserve:faults.append("acquisition_or_free_space_postflight_boundary_hit")
 return finish(cfg,path,bound,sorted(set(faults)),True,receipt,probe,free_before,observed,cache,combined)

def finish(cfg,path,bound,faults,executed,receipt,probe,free_before,observed=None,cache=None,combined=0):
 green=executed and not faults;target=bound.get("target")
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if green else ("PAUSED" if not faults else "RED"),"state":"EXACT_PROJECT_LOCAL_PYTHON_TOOLCHAIN_QUALIFIED" if green else ("READY_FOR_EXACT_PROJECT_LOCAL_PYTHON_ACQUISITION" if not executed and not faults else "PYTHON_TOOLCHAIN_ACQUISITION_SCOPED_DISPOSITION"),"faults":faults,"config":{"path":p2a.rel(path),"sha256":p2a.sha256_file(path)},"execution_performed":executed,"command":receipt and receipt.get("command"),"receipt":receipt,"version_probe":probe,"interpreter":{"path":p2a.rel(target) if target else "","sha256":p2a.sha256_file(target) if target and target.is_file() else "","version":cfg.get("version") if green else ""},"installed_tree":observed or {},"cache":cache or {},"combined_acquisition_and_cache_bytes":combined,"free_bytes_before":free_before,"free_bytes_after":shutil.disk_usage(ROOT).free,"network_downloads":1 if executed else 0,"version_probe_executions":1 if probe else 0,"global_bin_registration_performed":False,"package_installations":0,"project_installations":0,"repository_executions":0,"parent_target_or_evaluator_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0,"teacher_calls":0,"project_selected_output_cap":None,"maximum_inference":cfg.get("maximum_inference")}
if __name__=="__main__":raise SystemExit(main())
