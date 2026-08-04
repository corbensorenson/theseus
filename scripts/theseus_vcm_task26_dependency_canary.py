#!/usr/bin/env python3
"""Qualify task 26's single identical uv dependency closure."""
from __future__ import annotations
import argparse,email.parser,hashlib,json,os,shutil,sys,tarfile,tempfile,tomllib
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_dependency_prefetch_canary as base  # noqa:E402
import theseus_vcm_dependency_prefetch_canary_v2 as bounded  # noqa:E402
POLICY="project_theseus_vcm_task26_dependency_canary_v1";STATE="PROSPECTIVE_TASK_26_EXACT_UV_DEPENDENCY_SYNC_AND_OFFLINE_REPLAY";DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_task26_dependency_canary.json"
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));ap.add_argument("--out",default="");ap.add_argument("--execute",action="store_true");a=ap.parse_args();p=p2a.resolve(a.config);cfg=p2a.read_json(p);r=execute(cfg,p) if a.execute else preflight(cfg,p);p2a.write_json(p2a.resolve(a.out or cfg["report"]),r);print(json.dumps(summary(r),indent=2,sort_keys=True));return 0 if r["trigger_state"] in {"GREEN","PAUSED"} else 2
def preflight(cfg:dict[str,Any],path:Path)->dict[str,Any]:
 faults=[]
 if cfg.get("policy")!=POLICY or cfg.get("state")!=STATE:faults.append("policy_or_state_invalid")
 owner=p2a.resolve(str(cfg.get("owner") or ""))
 if owner!=Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner)!=cfg.get("owner_sha256"):faults.append("owner_binding_invalid")
 reports={}
 for name,raw in p2a.mapping(cfg.get("reports")).items():
  b=p2a.mapping(raw);p=p2a.resolve(str(b.get("path") or ""))
  if not p.is_file() or p2a.sha256_file(p)!=b.get("sha256"):faults.append(f"report_binding_invalid:{name}");reports[name]={}
  else:reports[name]=p2a.read_json(p)
 row=next((r for r in p2a.dicts(reports.get("prefetch_plan",{}).get("schedule")) if r.get("schedule_ordinal")==6),{});compat=next((r for r in p2a.dicts(reports.get("compatibility_v3",{}).get("rows")) if r.get("index")==26),{});pair=next((r for r in p2a.dicts(reports.get("pair_coverage",{}).get("rows")) if r.get("index")==26),{})
 if row.get("index")!=26 or row.get("manager")!="uv":faults.append("schedule_binding_invalid")
 if compat.get("state")!="COMPATIBLE_DECLARED_REQUIREMENTS" or "uv0_11_28_python3_12_5" not in compat.get("compatible_profile_ids",[]):faults.append("compatibility_binding_invalid")
 if pair.get("state")!="IDENTICAL_PARENT_TARGET_DEPENDENCY_IDENTITY" or pair.get("required_distinct_closure_count")!=1:faults.append("pair_coverage_binding_invalid")
 if reports.get("task30_closure_audit",{}).get("trigger_state")!="GREEN":faults.append("predecessor_boundary_invalid")
 observed={}
 for label,raw in p2a.mapping(cfg.get("archives")).items():
  b=p2a.mapping(raw);p=p2a.resolve(str(b.get("path") or ""))
  if not p.is_file() or p2a.sha256_file(p)!=b.get("sha256"):faults.append(f"archive_binding_invalid:{label}");continue
  ids,errs=base.archive_lock_identities(p,str(b.get("archive_root") or ""),cfg);faults.extend(f"{label}:{e}" for e in errs);packages=lock_packages(p,str(b.get("archive_root") or ""));observed[label]={"dependency_identity":ids,"package_count":len(packages),"artifact_identity_sha256":digest(packages)}
 if observed.get("parent")!=observed.get("target"):faults.append("parent_target_dependency_identity_mismatch")
 task=p2a.mapping(cfg.get("task"))
 if p2a.mapping(observed.get("target")).get("package_count")!=task.get("lock_package_count") or p2a.mapping(observed.get("target")).get("artifact_identity_sha256")!=task.get("lock_artifact_identity_sha256"):faults.append("lock_denominator_invalid")
 for name,raw in p2a.mapping(cfg.get("tools")).items():
  b=p2a.mapping(raw);p=p2a.resolve(str(b.get("path") or ""))
  if not p.is_file() or p2a.sha256_file(p)!=b.get("sha256"):faults.append(f"tool_binding_invalid:{name}")
 online=p2a.strings(p2a.mapping(cfg.get("commands")).get("online_sync_args"));offline=p2a.strings(p2a.mapping(cfg.get("commands")).get("offline_replay_args"));required=["sync","--frozen","--no-build","--no-install-project","--no-install-workspace","--no-default-groups","--no-python-downloads","--link-mode","copy","--color","never","--no-progress"]
 if online!=required or offline!=[*required,"--offline"]:faults.append("command_contract_invalid")
 allowed={"temporary_normalized_archive_extraction_authorized","single_network_dependency_sync_authorized","network_denied_offline_replay_authorized","wheel_only_installation_authorized","content_addressed_cache_retention_authorized"}
 for k,v in p2a.mapping(cfg.get("authority")).items():
  if v is not (k in allowed):faults.append(f"authority_invalid:{k}")
 store=p2a.resolve(str(cfg.get("retained_store") or ""))
 if store.parent!=(ROOT/"runtime/vcm_evaluator/dependency_store/uv").resolve() or store.name!="task-26":faults.append("retained_store_invalid")
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"RED" if faults else "PAUSED","state":"CONTRACT_INVALID" if faults else "READY_FOR_TASK_26_EXACT_UV_DEPENDENCY_CANARY","faults":sorted(set(faults)),"config":base.identity(path),"observed_parent_target_dependency_identities":observed,"execution_performed":False,"network_enabled_dependency_syncs":0,"network_denied_offline_replays":0,"source_build_executions":0,"project_installations":0,"repository_runner_executions":0,"parent_target_or_evaluator_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0,"maximum_inference":cfg.get("maximum_inference")}
def execute(cfg:dict[str,Any],path:Path)->dict[str,Any]:
 before=preflight(cfg,path)
 if before["trigger_state"]=="RED":return before
 store=p2a.resolve(str(cfg["retained_store"]));limits=p2a.mapping(cfg["limits"])
 if store.exists():return finish(before,["retained_store_already_exists"],{},False)
 free=shutil.disk_usage(ROOT).free
 if free<int(limits["minimum_free_bytes_after_execution"]):return finish(before,["free_space_reserve_preflight_boundary_hit"],{"free_bytes_before":free},False)
 faults=[];receipts={"free_bytes_before":free}
 with tempfile.TemporaryDirectory(prefix="theseus-vcm-task26-deps-",dir="/private/tmp") as raw:
  work=Path(raw).resolve();repo=work/"repository";cache=work/"cache";home=work/"home";tmp=work/"tmp";cache.mkdir();home.mkdir();tmp.mkdir();target=p2a.mapping(p2a.mapping(cfg["archives"])["target"]);receipts["repository_extraction"],errs=base.safe_extract_repository(p2a.resolve(str(target["path"])),repo,str(target["archive_root"]));faults.extend(errs);source_before=base.tree_identity(repo,excluded_roots={".venv"}) if not errs else {}
  tools=p2a.mapping(cfg["tools"]);python=str(p2a.resolve(str(p2a.mapping(tools["python"])["path"])));env={"HOME":str(home),"TMPDIR":str(tmp),"PATH":"/usr/bin:/bin","CI":"1","NO_COLOR":"1","UV_PYTHON":python,"UV_PYTHON_DOWNLOADS":"never"}
  if not faults:receipts["online_sync"]=bounded.run_sandboxed(uv_command(cfg,p2a.strings(p2a.mapping(cfg["commands"])["online_sync_args"]),python,cache),repo,work,env,cfg,network_denied=False);faults.extend(["online_sync_failed"] if base.command_failed(receipts["online_sync"]) else [])
  receipts["online_environment"]=inspect_environment(repo/".venv",cfg,target) if not faults else {}
  if not faults:faults.extend(f"online:{e}" for e in receipts["online_environment"]["faults"])
  if (repo/".venv").exists():shutil.rmtree(repo/".venv")
  if not faults:receipts["offline_replay"]=bounded.run_sandboxed(uv_command(cfg,p2a.strings(p2a.mapping(cfg["commands"])["offline_replay_args"]),python,cache),repo,work,env,cfg,network_denied=True);faults.extend(["offline_replay_failed"] if base.command_failed(receipts["offline_replay"]) else [])
  receipts["offline_environment"]=inspect_environment(repo/".venv",cfg,target) if not faults else {}
  if not faults:faults.extend(f"offline:{e}" for e in receipts["offline_environment"]["faults"])
  if not faults and receipts["online_environment"].get("distributions")!=receipts["offline_environment"].get("distributions"):faults.append("online_offline_environment_mismatch")
  source_after=base.tree_identity(repo,excluded_roots={".venv"}) if repo.exists() else {};receipts["repository_source_before"]=source_before;receipts["repository_source_after"]=source_after
  if source_before!=source_after:faults.append("repository_source_mutated_outside_venv")
  receipts["cache"]=tree_identity(cache);retained=int(receipts["cache"]["bytes"])
  if retained>int(limits["maximum_retained_bytes"]):faults.append("retained_store_size_boundary_hit")
  if shutil.disk_usage(ROOT).free-retained<int(limits["minimum_free_bytes_after_execution"]):faults.append("free_space_reserve_postflight_boundary_hit")
  if not faults:store.parent.mkdir(parents=True,exist_ok=True);os.replace(cache,store)
 receipts["retained_store"]=tree_identity(store) if store.exists() else {};receipts["free_bytes_after"]=shutil.disk_usage(ROOT).free;return finish(before,faults,receipts,True)
def uv_command(cfg:dict[str,Any],args:list[str],python:str,cache:Path)->list[str]:return [str(p2a.resolve(str(p2a.mapping(p2a.mapping(cfg["tools"])["uv"])["path"]))),*args,"--python",python,"--cache-dir",str(cache)]
def inspect_environment(venv:Path,cfg:dict[str,Any],side:dict[str,Any])->dict[str,Any]:
 parser=email.parser.Parser();rows=[];faults=[];site=venv/"lib/python3.12/site-packages"
 for meta in sorted(site.glob("*.dist-info/METADATA")) if site.is_dir() else []:
  m=parser.parsestr(meta.read_text(errors="replace"));name=normal(str(m.get("Name") or ""));version=str(m.get("Version") or "")
  if name and version:rows.append({"name":name,"version":version,"metadata_sha256":p2a.sha256_file(meta)})
 locked={(normal(str(r.get("name") or "")),str(r.get("version") or "")) for r in lock_packages(p2a.resolve(str(side["path"])),str(side["archive_root"]))}
 for row in rows:
  if (row["name"],row["version"]) not in locked:faults.append(f"installed_distribution_not_locked:{row['name']}@{row['version']}")
 names={r["name"] for r in rows}
 for required in p2a.strings(p2a.mapping(cfg.get("task")).get("required_runtime_dependencies")):
  if required not in names:faults.append(f"required_runtime_dependency_missing:{required}")
 if "quodeq" in names:faults.append("project_was_installed")
 return {"distribution_count":len(rows),"distributions":rows,"identity_sha256":base.digest_json(rows),"faults":sorted(set(faults))}
def lock_packages(archive:Path,root:str)->list[dict[str,Any]]:
 with tarfile.open(archive,"r:gz") as h:e=h.extractfile(root+"/uv.lock");v=tomllib.loads((e.read() if e else b"").decode())
 return sorted([{"name":x.get("name"),"version":x.get("version"),"source":x.get("source"),"sdist":x.get("sdist"),"wheels":x.get("wheels",[])} for x in v.get("package",[])],key=lambda r:(str(r["name"]),str(r["version"])))
def tree_identity(root:Path)->dict[str,Any]:
 files=[p for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()] if root.exists() else [];rows=[{"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":p2a.sha256_file(p)} for p in files];return {"path":p2a.rel(root),"file_count":len(files),"bytes":sum(p.stat().st_size for p in files),"identity_sha256":base.digest_json(rows)}
def digest(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def normal(v:str)->str:return v.lower().replace("_","-")
def finish(before:dict[str,Any],faults:list[str],receipts:dict[str,Any],executed:bool)->dict[str,Any]:return {**before,"created_utc":p2a.now(),"trigger_state":"GREEN" if executed and not faults else "RED","state":"TASK_26_EXACT_UV_DEPENDENCY_CACHE_ACQUIRED_AND_OFFLINE_REPLAY_QUALIFIED" if executed and not faults else "TASK_26_DEPENDENCY_CANARY_FAILED","faults":sorted(set(faults)),"execution_performed":executed,"network_enabled_dependency_syncs":int("online_sync" in receipts),"network_denied_offline_replays":int("offline_replay" in receipts),"receipts":receipts}
def summary(r:dict[str,Any])->dict[str,Any]:return {k:r.get(k) for k in ("trigger_state","state","execution_performed","network_enabled_dependency_syncs","network_denied_offline_replays","source_build_executions","project_installations","repository_runner_executions","parent_target_or_evaluator_executions","candidate_or_control_calls","external_reference_calls","faults")}
if __name__=="__main__":raise SystemExit(main())
