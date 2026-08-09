#!/usr/bin/env python3
"""Resolve exact immutable locks for the three host-adequate VCM replacements."""

from __future__ import annotations
import argparse,hashlib,json,os,shutil,sys,tarfile,tempfile
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_immutable_resolution_segment as prior  # noqa:E402
POLICY="project_theseus_vcm_replacement_resolution_v3";DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_replacement_resolution.json";EXPECTED=[12,13,35]

def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));p.add_argument("--out",default="");p.add_argument("--execute",action="store_true");a=p.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=execute(path) if a.execute else preflight_report(path);p2a.write_json(p2a.resolve(a.out or cfg["report"]),r);print(json.dumps(summary(r),indent=2,sort_keys=True));return 0 if r["trigger_state"] in {"GREEN","PAUSED"} else 2

def preflight(path:Path=DEFAULT_CONFIG)->tuple[dict[str,Any],dict[str,Any],list[str]]:
 cfg=p2a.read_json(path);faults=[]
 if cfg.get("policy")!=POLICY:faults.append("policy_invalid")
 for b in p2a.dicts(cfg.get("bindings")):
  p=p2a.resolve(str(b.get("path") or ""))
  if not p.is_file() or p2a.sha256_file(p)!=b.get("sha256"):faults.append(f"binding_invalid:{b.get('id')}")
 panel_path=p2a.resolve(cfg["source_panel"]);closure_path=p2a.resolve(cfg["repository_closures"]);panel=p2a.read_json(panel_path) if panel_path.is_file() else {};closures=p2a.read_json(closure_path) if closure_path.is_file() else {}
 if panel.get("trigger_state")!="GREEN" or panel.get("source_panel_admitted") is not True:faults.append("source_panel_invalid")
 if closures.get("trigger_state")!="GREEN" or closures.get("archive_artifacts")!=124:faults.append("closure_report_invalid")
 panel_rows={int(r.get("index") or 0):r for r in p2a.dicts(panel.get("assembled_rows"))};closure_rows={int(r.get("campaign_index") or 0):r for r in p2a.dicts(closures.get("tasks"))};rows=p2a.dicts(cfg.get("rows"));indices=[int(r.get("index") or 0) for r in rows]
 if indices!=EXPECTED or len(set(indices))!=3:faults.append("resolution_denominator_invalid")
 bound={}
 for row in rows:
  index=int(row["index"]);panel_row=panel_rows.get(index,{});closure_row=closure_rows.get(index,{});target=next((r for r in p2a.dicts(closure_row.get("artifacts")) if r.get("label")=="target"),{});archive=p2a.resolve(str(target.get("normalized") or ""));root=str(target.get("source_archive_root") or "")
  if row.get("repository")!=panel_row.get("repository") or row.get("repository")!=closure_row.get("repository"):faults.append(f"task_{index}:source_alignment_invalid")
  if not archive.is_file() or p2a.sha256_file(archive)!=target.get("normalized_sha256"):faults.append(f"task_{index}:target_archive_invalid")
  manifests=[]
  if archive.is_file():
   with tarfile.open(archive,"r:gz") as h:
    for rec in p2a.dicts(row.get("manifest_receipts")):
     rel=str(rec.get("path") or "")
     try:payload=h.extractfile(h.getmember(f"{root}/{rel}")).read()
     except (KeyError,AttributeError):faults.append(f"task_{index}:manifest_missing:{rel}");continue
     observed={"path":rel,"sha256":hashlib.sha256(payload).hexdigest()};manifests.append(observed)
     if observed!=rec:faults.append(f"task_{index}:manifest_identity_invalid:{rel}")
  if row.get("manager") not in {"uv","cargo_static","receipt_reuse"}:faults.append(f"task_{index}:manager_invalid")
  if row.get("manager")=="uv" and str(row.get("python_tool") or "") not in p2a.mapping(cfg.get("tools")):faults.append(f"task_{index}:python_tool_invalid")
  reuse_payload=b""
  if row.get("manager")=="receipt_reuse":
   source_report=p2a.resolve(str(row.get("source_report") or ""));source=p2a.read_json(source_report) if source_report.is_file() else {};source_row=next((r for r in p2a.dicts(source.get("rows")) if int(r.get("index") or 0)==index),{});receipt=p2a.mapping(source_row.get("receipt"));lock=p2a.mapping(receipt.get("lock"));reuse_payload=str(receipt.get("stdout") or "").encode();expected=p2a.mapping(row.get("expected_lock"))
   if source.get("trigger_state")!="PAUSED" or source_row.get("disposition")!="RESOLUTION_QUALIFIED_IMMUTABLE_LOCK_STAGED":faults.append(f"task_{index}:reuse_source_disposition_invalid")
   if hashlib.sha256(reuse_payload).hexdigest()!=expected.get("sha256") or len(reuse_payload)!=expected.get("bytes") or lock.get("sha256")!=expected.get("sha256") or lock.get("bytes")!=expected.get("bytes"):faults.append(f"task_{index}:reuse_lock_identity_invalid")
  bound[index]={"row":row,"target":target,"archive":archive,"root":root,"manifests":manifests,"reuse_payload":reuse_payload}
 authority=p2a.mapping(cfg.get("authority"));allowed={"target_archive_extraction_authorized","registry_metadata_resolution_authorized","immutable_lock_write_authorized","shared_resolver_cache_authorized","repository_owned_static_lock_copy_authorized","sealed_receipt_lock_reuse_authorized"}
 if any(authority.get(k) is not True for k in allowed) or any(v is not False for k,v in authority.items() if k not in allowed):faults.append("authority_invalid")
 outdir=p2a.resolve(str(cfg.get("output_directory") or ""));cache=p2a.resolve(str(cfg.get("shared_cache_root") or ""))
 if outdir!=(ROOT/"reports/theseus_vcm_replacement_resolution_locks").resolve():faults.append("output_directory_invalid")
 if cache!=(ROOT/"runtime/vcm_evaluator/dependency_store/replacement-resolution-v3").resolve():faults.append("shared_cache_root_invalid")
 return cfg,{"panel":panel,"closures":closures,"rows":bound},sorted(set(faults))

def preflight_report(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg,bound,faults=preflight(path);return finish(cfg,path,[],faults,False,None,shutil.disk_usage(ROOT).free)

def execute(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg,bound,faults=preflight(path);free_before=shutil.disk_usage(ROOT).free;limits=p2a.mapping(cfg.get("limits"));reserve=int(limits["minimum_free_bytes_after_execution"]);maximum=int(limits["maximum_shared_cache_bytes"]);cache=p2a.resolve(cfg["shared_cache_root"]);outdir=p2a.resolve(cfg["output_directory"])
 if faults:return finish(cfg,path,[],faults,False,cache,free_before)
 if free_before-maximum<reserve:return finish(cfg,path,[],["free_space_reserve_preflight_boundary_hit"],False,cache,free_before)
 if outdir.exists():return finish(cfg,path,[],["immutable_output_directory_already_exists"],False,cache,free_before)
 cache.mkdir(parents=True,exist_ok=True);outdir.parent.mkdir(parents=True,exist_ok=True);results=[];publish=Path(tempfile.mkdtemp(prefix=".theseus-vcm-replacement-resolution-",dir=outdir.parent))
 with tempfile.TemporaryDirectory(prefix="theseus-vcm-replacement-resolution-",dir="/private/tmp") as raw:
  tmp=Path(raw)
  for index in EXPECTED:
   item=bound["rows"][index];row=p2a.mapping(item["row"]);work=tmp/f"task-{index:02d}";root,extract_faults=prior.extract_regular_archive(item["archive"],work)
   if extract_faults:results.append(failure(row,item,"INCONCLUSIVE_IMPLEMENTATION_ARCHIVE_EXTRACTION",extract_faults));continue
   before=prior.tree_identity(root)
   if row["manager"]=="receipt_reuse":
    generated=tmp/f"task-{index:02d}-reused.lock";generated.write_bytes(item["reuse_payload"]);command=[];receipt={"returncode":0,"duration_ms":0.0,"boundary_hit":False,"stdout":"","stderr":"","stdout_complete":True,"stderr_complete":True,"project_selected_output_cap":None,"sealed_receipt_lock_reuse":True,"source_report":row["source_report"]}
   elif row["manager"]=="cargo_static":
    source=root/"Cargo.lock";command=[];receipt={"returncode":0,"duration_ms":0.0,"boundary_hit":False,"stdout":"","stderr":"","stdout_complete":True,"stderr_complete":True,"project_selected_output_cap":None,"static_repository_lock_copy":True}
    if not source.is_file():results.append(failure(row,item,"INCONCLUSIVE_IMPLEMENTATION_LOCK_OUTPUT_MISSING",["repository_cargo_lock_missing"]));continue
    generated=source
   else:
    generated=tmp/f"task-{index:02d}-requirements.lock";python=str(p2a.resolve(p2a.mapping(cfg["tools"])[str(row["python_tool"])]["path"]));uv=str(p2a.resolve(p2a.mapping(cfg["tools"])["uv"]["path"]));home=tmp/f"home-{index:02d}";temp=tmp/f"tmp-{index:02d}";home.mkdir();temp.mkdir();env={"HOME":str(home),"TMPDIR":str(temp),"PATH":"/usr/bin:/bin","CI":"1","NO_COLOR":"1","LANG":"C","LC_ALL":"C","UV_PYTHON":python,"UV_PYTHON_DOWNLOADS":"never","UV_NO_CONFIG":"1"};command=[uv,"pip","compile",str(root/row["input"])]
    for extra in p2a.strings(row.get("extras")):command.extend(["--extra",extra])
    for group in p2a.strings(row.get("groups")):command.extend(["--group",group])
    command.extend(["--python",python,"--python-platform","aarch64-apple-darwin","--generate-hashes","--no-build","--index-strategy","first-index","--default-index","https://pypi.org/simple","--cache-dir",str(cache/"uv"),"--output-file",str(generated),"--color","never","--no-progress"])
    try:receipt=prior.run(command,root,env,limits)
    except OSError as exc:receipt={"returncode":None,"duration_ms":0.0,"boundary_hit":False,"stdout":"","stderr":"","stdout_complete":True,"stderr_complete":True,"project_selected_output_cap":None,"spawn_error":{"type":type(exc).__name__,"message":str(exc)}}
   after=prior.tree_identity(root);receipt.update({"source_before":before,"source_after":after});row_faults=[]
   if before!=after:row_faults.append("source_mutated_during_resolution")
   if receipt.get("boundary_hit"):disposition="INCONCLUSIVE_EXPERIMENT_HOST_RESOURCE_BOUNDARY"
   elif receipt.get("spawn_error"):disposition="INCONCLUSIVE_IMPLEMENTATION_RESOLVER_SPAWN"
   elif receipt.get("returncode")!=0:disposition="INCONCLUSIVE_EXPERIMENT_DEPENDENCY_RESOLUTION"
   elif not generated.is_file():disposition="INCONCLUSIVE_IMPLEMENTATION_LOCK_OUTPUT_MISSING";row_faults.append("generated_lock_missing")
   else:
    manager="cargo" if row["manager"]=="cargo_static" else "uv";lock,lock_faults=prior.validate_lock(manager,generated);row_faults.extend(lock_faults)
    if lock_faults:disposition="INCONCLUSIVE_IMPLEMENTATION_LOCK_VALIDATION"
    else:
     staged=publish/str(row["output_name"]);shutil.copyfile(generated,staged);lock.update({"sha256":p2a.sha256_file(staged),"bytes":staged.stat().st_size});receipt["lock"]=lock;disposition="RESOLUTION_QUALIFIED_IMMUTABLE_LOCK_STAGED"
   results.append({"index":index,"repository":row["repository"],"manager":row["manager"],"target_archive":p2a.rel(item["archive"]),"target_archive_sha256":p2a.sha256_file(item["archive"]),"command":command,"receipt":receipt,"faults":sorted(set(row_faults)),"disposition":disposition})
   if prior.tree_identity(cache).get("bytes",0)>maximum or shutil.disk_usage(ROOT).free<reserve:faults.append("cache_or_free_space_postflight_boundary_hit");break
  staged_qualified=len(results)==3 and not faults and all(r.get("disposition")=="RESOLUTION_QUALIFIED_IMMUTABLE_LOCK_STAGED" for r in results)
  if staged_qualified:
   os.replace(publish,outdir)
   for result in results:
    lock=p2a.mapping(p2a.mapping(result["receipt"])["lock"]);row=next(r for r in p2a.dicts(cfg["rows"]) if int(r["index"])==int(result["index"]));output=outdir/str(row["output_name"]);lock.update({"path":p2a.rel(output),"sha256":p2a.sha256_file(output),"bytes":output.stat().st_size});result["disposition"]="RESOLUTION_QUALIFIED_IMMUTABLE_LOCK"
  elif len(results)!=3 and not faults:
   faults.append("replacement_resolution_denominator_not_completed")
  if publish.exists():shutil.rmtree(publish)
 return finish(cfg,path,results,faults,True,cache,free_before)

def failure(row,item,disposition,faults):return {"index":row["index"],"repository":row["repository"],"manager":row["manager"],"target_archive":p2a.rel(item["archive"]),"target_archive_sha256":p2a.sha256_file(item["archive"]),"command":[],"receipt":{},"faults":sorted(set(faults)),"disposition":disposition}
def finish(cfg,path,rows,faults,executed,cache,free_before):
 qualified=sum(r.get("disposition")=="RESOLUTION_QUALIFIED_IMMUTABLE_LOCK" for r in rows);green=executed and not faults and qualified==3
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if green else ("PAUSED" if not faults else "RED"),"state":"THREE_REPLACEMENT_LOCKS_RESOLUTION_QUALIFIED" if green else ("READY_FOR_THREE_REPLACEMENT_RESOLUTIONS" if not executed and not faults else "REPLACEMENT_RESOLUTION_SCOPED_DISPOSITIONS"),"faults":sorted(set(faults)),"config":{"path":p2a.rel(path),"sha256":p2a.sha256_file(path)},"predecessor_invalidated_attempt":cfg.get("predecessor_invalidated_attempt"),"execution_performed":executed,"task_count":len(rows),"qualified_task_count":qualified,"inconclusive_task_count":len(rows)-qualified,"network_resolution_task_count":sum(r.get("manager")=="uv" for r in rows),"sealed_receipt_reuse_task_count":sum(r.get("manager")=="receipt_reuse" for r in rows),"static_lock_task_count":sum(r.get("manager")=="cargo_static" for r in rows),"rows":rows,"cache":prior.tree_identity(cache) if cache else {},"free_bytes_before":free_before,"free_bytes_after":shutil.disk_usage(ROOT).free,"panel_admitted":False,"partial_panel_admission_forbidden":True,"package_installations":0,"source_build_executions":0,"repository_runner_executions":0,"parent_target_or_evaluator_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0,"teacher_calls":0,"project_selected_output_cap":None,"maximum_inference":cfg.get("maximum_inference")}
def summary(r):return {k:r.get(k) for k in ("trigger_state","state","faults","execution_performed","task_count","qualified_task_count","inconclusive_task_count","package_installations","source_build_executions","parent_target_or_evaluator_executions","candidate_or_control_calls","external_reference_calls")}
if __name__=="__main__":raise SystemExit(main())
