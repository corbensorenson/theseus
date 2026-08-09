#!/usr/bin/env python3
"""Build a sealed manifest of prequalified sdists under network-denied sandboxes."""
from __future__ import annotations
import argparse,email.parser,hashlib,json,os,re,shutil,subprocess,sys,tempfile,time,zipfile
from pathlib import Path,PurePosixPath
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
POLICY="project_theseus_vcm_sdist_batch_build_v1";STATE="PROSPECTIVE_TASK13_NETWORK_DENIED_TRANSITIVE_SDIST_BATCH_BUILD_V1";DEFAULT_CONFIG=ROOT/"configs/theseus_vcm_sdist_batch_build.json"
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));ap.add_argument("--out",default="");ap.add_argument("--execute",action="store_true");a=ap.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=execute(path) if a.execute else preflight_report(path);p2a.write_json(p2a.resolve(a.out or cfg["report"]),r);print(json.dumps({k:r.get(k) for k in ("trigger_state","state","faults","execution_performed","row_count","qualified_row_count","inconclusive_row_count","source_build_executions","candidate_or_control_calls","external_reference_calls")},indent=2,sort_keys=True));return 0 if r["trigger_state"] in {"GREEN","PAUSED"} else 2
def preflight(path:Path=DEFAULT_CONFIG)->tuple[dict[str,Any],dict[str,Any],list[str]]:
 cfg=p2a.read_json(path);faults=[]
 if cfg.get("policy")!=POLICY or cfg.get("state")!=STATE:faults.append("policy_or_state_invalid")
 for key,expected in (("owner",Path(__file__).resolve()),("audit_owner",ROOT/"scripts/theseus_vcm_sdist_batch_build_audit.py")):
  owner=p2a.resolve(str(cfg.get(key) or ""))
  if owner!=expected.resolve() or not owner.is_file() or p2a.sha256_file(owner)!=cfg.get(f"{key}_sha256"):faults.append(f"{key}_binding_invalid")
 sources={}
 for name,raw in p2a.mapping(cfg.get("sources")).items():
  b=p2a.mapping(raw);p=p2a.resolve(str(b.get("path") or ""))
  if not p.is_file() or p2a.sha256_file(p)!=b.get("sha256"):faults.append(f"source_binding_invalid:{name}");sources[name]={}
  else:sources[name]=p2a.read_json(p)
 batch=sources.get("sdist_batch",{});audit=sources.get("sdist_batch_audit",{})
 if batch.get("trigger_state")!="GREEN" or batch.get("eligible_row_count")!=3 or batch.get("inconclusive_row_count")!=0:faults.append("sdist_batch_not_fully_eligible")
 if audit.get("trigger_state")!="GREEN" or audit.get("eligible_row_count")!=3:faults.append("sdist_batch_audit_not_green")
 tools={}
 for name,raw in p2a.mapping(cfg.get("tools")).items():
  b=p2a.mapping(raw);tool=p2a.resolve(str(b.get("path") or ""))
  if not tool.is_file() or p2a.sha256_file(tool)!=b.get("sha256"):faults.append(f"tool_binding_invalid:{name}")
  tools[name]=str(tool)
 source_rows={(r.get("name"),r.get("version")):r for r in p2a.dicts(batch.get("rows"))};rows=p2a.dicts(cfg.get("rows"));expected=[("habluetooth","6.26.5"),("bluetooth-data-tools","1.29.21"),("PyRIC","0.1.6.3")]
 if [(r.get("name"),r.get("version")) for r in rows]!=expected:faults.append("build_batch_denominator_invalid")
 bound_rows={}
 for row in rows:
  key=(row.get("name"),row.get("version"));source=source_rows.get(key,{});retained=p2a.mapping(source.get("retained_sdist"));sdist=p2a.resolve(str(retained.get("path") or ""))
  if source.get("disposition")!="ELIGIBLE_FOR_NETWORK_DENIED_BATCH_BUILD" or not sdist.is_file() or p2a.sha256_file(sdist)!=retained.get("sha256"):faults.append(f"sdist_binding_invalid:{key[0]}")
  declared={requirement_name(v) for v in p2a.strings(p2a.mapping(source.get("inspection")).get("build_requirements"))};pinned=p2a.strings(row.get("pinned_build_requirements"));pinned_names={requirement_name(v) for v in pinned}
  if not declared or declared!=pinned_names or any("==" not in v for v in pinned):faults.append(f"pinned_build_profile_invalid:{key[0]}")
  bound_rows[key]={"config":row,"source":source,"sdist":sdist}
 store=p2a.resolve(str(cfg.get("wheel_store") or ""))
 if store!=(ROOT/"runtime/vcm_evaluator/dependency_store/wheels").resolve():faults.append("wheel_store_invalid")
 allowed={"disposable_pinned_build_environment_authorized","network_build_tool_wheel_sync_authorized","exact_prequalified_sdist_build_authorized","network_denied_sandbox_build_authorized","built_wheel_retention_authorized"}
 for key,value in p2a.mapping(cfg.get("authority")).items():
  if value is not (key in allowed):faults.append(f"authority_invalid:{key}")
 return cfg,{"sources":sources,"tools":tools,"rows":bound_rows,"store":store},sorted(set(faults))
def requirement_name(value:str)->str:return re.split(r"[<>=!~ ;\[]",value,1)[0].lower().replace("_","-")
def preflight_report(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg,_,faults=preflight(path);return finish(cfg,path,faults,[],False)
def execute(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg,bound,faults=preflight(path)
 if faults:return finish(cfg,path,faults,[],False)
 limits=p2a.mapping(cfg["limits"])
 if shutil.disk_usage(ROOT).free-int(limits["maximum_temporary_bytes"])<int(limits["minimum_free_bytes_after_execution"]):return finish(cfg,path,["free_space_reserve_preflight_boundary_hit"],[],False)
 results=[]
 with tempfile.TemporaryDirectory(prefix="theseus-vcm-sdist-batch-build-",dir="/private/tmp") as raw:
  root=Path(raw).resolve();cache=root/"shared-cache";cache.mkdir()
  for ordinal,(key,item) in enumerate(bound["rows"].items(),1):
   row=p2a.mapping(item["config"]);work=root/f"row-{ordinal:02d}";venv=work/"venv";out=work/"out";home=work/"home";tmp=work/"tmp"
   for directory in (work,out,home,tmp):directory.mkdir(exist_ok=True)
   env={"HOME":str(home),"TMPDIR":str(tmp),"PATH":"/usr/bin:/bin","LANG":"C","LC_ALL":"C","CI":"1","NO_COLOR":"1","UV_PYTHON_DOWNLOADS":"never","UV_NO_CONFIG":"1"};uv=bound["tools"]["uv"];python=bound["tools"]["python"];sandbox=bound["tools"]["sandbox_exec"];receipts={};row_faults=[]
   receipts["venv"]=run([uv,"venv",str(venv),"--python",python,"--no-python-downloads","--no-config"],work,env,limits)
   if receipts["venv"]["returncode"]!=0:row_faults.append("build_venv_failed")
   sync=[uv,"pip","install","--python",str(venv/"bin/python"),"--no-build","--default-index","https://pypi.org/simple","--cache-dir",str(cache),"--no-config","--no-progress","--color","never",*p2a.strings(row["pinned_build_requirements"])]
   if not row_faults:
    receipts["build_tool_sync"]=run(sync,work,env,limits)
    if receipts["build_tool_sync"]["returncode"]!=0:row_faults.append("pinned_build_tool_sync_failed")
   profile="\n".join(["(version 1)","(allow default)","(deny network*)",f'(deny file-write* (require-not (subpath "{work}")))','(allow file-write* (literal "/dev/null"))'])
   build=[sandbox,"-p",profile,uv,"build",str(item["sdist"]),"--wheel","--out-dir",str(out),"--python",str(venv/"bin/python"),"--no-build-isolation","--offline","--cache-dir",str(cache),"--no-config","--no-progress","--color","never"]
   if not row_faults:
    receipts["sandbox_build"]=run(build,work,env,limits)
    if receipts["sandbox_build"]["returncode"]!=0:row_faults.append("network_denied_sandbox_build_failed")
   wheels=sorted(out.glob("*.whl"))
   if not row_faults and len(wheels)!=1:row_faults.append("wheel_denominator_invalid")
   if not row_faults:
    wheel_receipt,wheel_faults=inspect_wheel(wheels[0],str(row["name"]),str(row["version"]));receipts["wheel"]=wheel_receipt;row_faults.extend(wheel_faults)
   if not row_faults:
    target:Path=bound["store"]/wheels[0].name;bound["store"].mkdir(parents=True,exist_ok=True)
    if target.exists():row_faults.append("retained_wheel_already_exists")
    else:shutil.copyfile(wheels[0],target);receipts["retained_wheel"]={"path":p2a.rel(target),"bytes":target.stat().st_size,"sha256":p2a.sha256_file(target)}
   disposition="QUALIFIED_NETWORK_DENIED_SDIST_WHEEL_BUILD" if not row_faults else ("INCONCLUSIVE_EXPERIMENT_HOST_RESOURCE_BOUNDARY" if any(p2a.mapping(v).get("boundary_hit") for v in receipts.values()) else "INCONCLUSIVE_IMPLEMENTATION_SDIST_BUILD")
   results.append({"name":row["name"],"version":row["version"],"pinned_build_requirements":row["pinned_build_requirements"],"sdist":{"path":p2a.rel(item["sdist"]),"sha256":p2a.sha256_file(item["sdist"])},"receipts":receipts,"faults":sorted(set(row_faults)),"disposition":disposition})
 return finish(cfg,path,faults,results,True)
def run(command:list[str],cwd:Path,env:dict[str,str],limits:dict[str,Any])->dict[str,Any]:
 start=time.monotonic()
 try:done=subprocess.run(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=int(limits["wall_seconds_per_command"]),check=False);boundary=False;reason=""
 except subprocess.TimeoutExpired as e:done=subprocess.CompletedProcess(command,124,e.stdout or b"",e.stderr or b"");boundary=True;reason="wall_timeout"
 stdout=done.stdout or b"";stderr=done.stderr or b""
 return {"command":command,"returncode":done.returncode,"duration_ms":round((time.monotonic()-start)*1000,3),"boundary_hit":boundary,"boundary_reason":reason,"stdout":stdout.decode("utf-8","replace"),"stderr":stderr.decode("utf-8","replace"),"stdout_bytes":len(stdout),"stderr_bytes":len(stderr),"stdout_sha256":hashlib.sha256(stdout).hexdigest(),"stderr_sha256":hashlib.sha256(stderr).hexdigest(),"stdout_complete":True,"stderr_complete":True,"project_selected_output_cap":None}
def inspect_wheel(path:Path,expected_name:str,expected_version:str)->tuple[dict[str,Any],list[str]]:
 faults=[];rows=[];metadata=b"";native=[]
 try:
  with zipfile.ZipFile(path) as wheel:
   for info in wheel.infolist():
    pure=PurePosixPath(info.filename)
    if pure.is_absolute() or ".." in pure.parts:faults.append("unsafe_wheel_member")
    payload=wheel.read(info);rows.append({"path":info.filename,"bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest()})
    if info.filename.endswith(".dist-info/METADATA"):metadata=payload
    if info.filename.endswith((".so",".dylib",".dll",".pyd")):native.append(info.filename)
 except (zipfile.BadZipFile,OSError):faults.append("wheel_parse_failed")
 parsed=email.parser.Parser().parsestr(metadata.decode("utf-8","replace")) if metadata else None;name=str(parsed.get("Name") or "") if parsed else "";version=str(parsed.get("Version") or "") if parsed else ""
 if name.lower().replace("_","-")!=expected_name.lower().replace("_","-") or version!=expected_version:faults.append("wheel_metadata_identity_invalid")
 ordered=sorted(rows,key=lambda r:r["path"])
 return {"filename":path.name,"bytes":path.stat().st_size if path.is_file() else 0,"sha256":p2a.sha256_file(path) if path.is_file() else "","member_count":len(rows),"member_receipts_sha256":hashlib.sha256(json.dumps(ordered,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"metadata_name":name,"metadata_version":version,"native_members":sorted(native)},sorted(set(faults))
def finish(cfg:dict[str,Any],path:Path,faults:list[str],rows:list[dict[str,Any]],execution:bool)->dict[str,Any]:
 qualified=sum(r.get("disposition")=="QUALIFIED_NETWORK_DENIED_SDIST_WHEEL_BUILD" for r in rows);attempts=sum("sandbox_build" in p2a.mapping(r.get("receipts")) for r in rows)
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"RED" if faults else ("GREEN" if execution else "PAUSED"),"state":"TASK13_NETWORK_DENIED_TRANSITIVE_SDIST_BATCH_BUILD_COMPLETE" if execution and not faults else ("READY_FOR_TASK13_NETWORK_DENIED_TRANSITIVE_SDIST_BATCH_BUILD" if not faults else "TASK13_NETWORK_DENIED_TRANSITIVE_SDIST_BATCH_BUILD_INVALID"),"faults":sorted(set(faults)),"config":{"path":p2a.rel(path),"sha256":p2a.sha256_file(path)},"execution_performed":execution,"row_count":len(rows),"qualified_row_count":qualified,"inconclusive_row_count":len(rows)-qualified,"rows":rows,"partial_batch_admitted":False,"source_build_executions":attempts,"network_denied_builds":attempts,"evaluator_package_installations":0,"repository_runner_executions":0,"parent_target_or_evaluator_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0,"teacher_calls":0,"next_authorized_boundary":"replay_task13_resolution_with_all_prequalified_local_wheels" if rows and qualified==len(rows) else "repair_inconclusive_batch_build_rows","maximum_inference":cfg.get("maximum_inference")}
if __name__=="__main__":raise SystemExit(main())
