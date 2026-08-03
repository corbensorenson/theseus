#!/usr/bin/env python3
"""Qualify separate parent and target uv dependency closures for VCM task 14."""
from __future__ import annotations
import argparse,email.parser,hashlib,json,os,shutil,sys,tarfile,tempfile,tomllib
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_dependency_prefetch_canary as base  # noqa:E402
import theseus_vcm_dependency_prefetch_canary_v2 as bounded  # noqa:E402
POLICY="project_theseus_vcm_task14_dual_dependency_canary_v1";STATE="PROSPECTIVE_TASK_14_SEPARATE_PARENT_TARGET_UV_CLOSURES";DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_task14_dual_dependency_canary.json"
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
 schedule=next((r for r in p2a.dicts(reports.get("prefetch_plan",{}).get("schedule")) if r.get("schedule_ordinal")==4),{});compat=next((r for r in p2a.dicts(reports.get("compatibility_v3",{}).get("rows")) if r.get("index")==14),{});pair=next((r for r in p2a.dicts(reports.get("pair_coverage",{}).get("rows")) if r.get("index")==14),{})
 if schedule.get("index")!=14 or schedule.get("manager")!="uv":faults.append("schedule_binding_invalid")
 if compat.get("state")!="COMPATIBLE_DECLARED_REQUIREMENTS" or "uv0_11_28_python3_12_5" not in compat.get("compatible_profile_ids",[]):faults.append("compatibility_binding_invalid")
 if pair.get("state")!="DIVERGENT_PARENT_TARGET_DEPENDENCY_IDENTITY" or pair.get("required_distinct_closure_count")!=2:faults.append("pair_coverage_binding_invalid")
 if reports.get("task36_closure_audit",{}).get("trigger_state")!="GREEN":faults.append("predecessor_boundary_invalid")
 observed={}
 for label,raw in p2a.mapping(cfg.get("sides")).items():
  side=p2a.mapping(raw);archive=p2a.resolve(str(side.get("archive") or ""))
  if not archive.is_file() or p2a.sha256_file(archive)!=side.get("archive_sha256"):faults.append(f"archive_binding_invalid:{label}");continue
  manifest=member_identity(archive,str(side.get("archive_root") or ""),"pyproject.toml");lock=member_identity(archive,str(side.get("archive_root") or ""),"uv.lock");packages=lock_packages(archive,str(side.get("archive_root") or ""));identity=hashlib.sha256(json.dumps(packages,sort_keys=True,separators=(",",":")).encode()).hexdigest();observed[label]={"manifest":manifest,"lock":lock,"package_count":len(packages),"artifact_identity_sha256":identity}
  if manifest.get("sha256")!=side.get("manifest_sha256") or lock.get("sha256")!=side.get("lock_sha256") or len(packages)!=side.get("lock_package_count") or identity!=side.get("lock_artifact_identity_sha256"):faults.append(f"side_dependency_binding_invalid:{label}")
 for name,raw in p2a.mapping(cfg.get("tools")).items():
  b=p2a.mapping(raw);p=p2a.resolve(str(b.get("path") or ""))
  if not p.is_file() or p2a.sha256_file(p)!=b.get("sha256"):faults.append(f"tool_binding_invalid:{name}")
 base_args=p2a.strings(p2a.mapping(cfg.get("commands")).get("online_sync_args"));offline=p2a.strings(p2a.mapping(cfg.get("commands")).get("offline_replay_args"));required=["sync","--frozen","--no-build","--no-install-project","--no-install-workspace","--no-default-groups","--no-python-downloads","--link-mode","copy","--color","never","--no-progress"]
 if base_args!=required or offline!=[*required,"--offline"]:faults.append("command_contract_invalid")
 allowed={"temporary_normalized_archive_extraction_authorized","separate_parent_target_network_sync_authorized","network_denied_offline_replay_authorized","wheel_only_installation_authorized","separate_cache_retention_authorized"}
 for k,v in p2a.mapping(cfg.get("authority")).items():
  if v is not (k in allowed):faults.append(f"authority_invalid:{k}")
 stores=[]
 for label,side in p2a.mapping(cfg.get("sides")).items():
  store=p2a.resolve(str(p2a.mapping(side).get("retained_store") or ""));stores.append(store)
  if store.parent!=(ROOT/"runtime/vcm_evaluator/dependency_store/uv").resolve() or store.name!=f"task-14-{label}":faults.append(f"retained_store_invalid:{label}")
 if len(set(stores))!=2:faults.append("retained_stores_not_distinct")
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"RED" if faults else "PAUSED","state":"CONTRACT_INVALID" if faults else "READY_FOR_TASK_14_DUAL_UV_DEPENDENCY_CANARY","faults":sorted(set(faults)),"config":base.identity(path),"observed_side_dependency_identities":observed,"execution_performed":False,"network_enabled_dependency_syncs":0,"network_denied_offline_replays":0,"source_build_executions":0,"project_installations":0,"repository_runner_executions":0,"parent_target_or_evaluator_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0,"maximum_inference":cfg.get("maximum_inference")}
def execute(cfg:dict[str,Any],path:Path)->dict[str,Any]:
 before=preflight(cfg,path)
 if before["trigger_state"]=="RED":return before
 if any(p2a.resolve(str(p2a.mapping(s).get("retained_store"))).exists() for s in p2a.mapping(cfg["sides"]).values()):return finish(before,["retained_store_already_exists"],{},False)
 limits=p2a.mapping(cfg["limits"]);free=shutil.disk_usage(ROOT).free
 if free<int(limits["minimum_free_bytes_after_execution"]):return finish(before,["free_space_reserve_preflight_boundary_hit"],{"free_bytes_before":free},False)
 faults=[];receipts={"free_bytes_before":free,"sides":{}}
 with tempfile.TemporaryDirectory(prefix="theseus-vcm-task14-dual-",dir="/private/tmp") as raw:
  work=Path(raw).resolve()
  for label,raw_side in p2a.mapping(cfg["sides"]).items():
   side=p2a.mapping(raw_side);side_root=work/label;repo=side_root/"repository";cache=side_root/"cache";home=side_root/"home";tmp=side_root/"tmp";cache.mkdir(parents=True);home.mkdir();tmp.mkdir();side_receipt={};receipts["sides"][label]=side_receipt
   side_receipt["repository_extraction"],errs=base.safe_extract_repository(p2a.resolve(str(side["archive"])),repo,str(side["archive_root"]));faults.extend(f"{label}:{e}" for e in errs);source_before=base.tree_identity(repo,excluded_roots={".venv"}) if not errs else {}
   tools=p2a.mapping(cfg["tools"]);python=str(p2a.resolve(str(p2a.mapping(tools["python"])["path"])));env={"HOME":str(home),"TMPDIR":str(tmp),"PATH":"/usr/bin:/bin","CI":"1","NO_COLOR":"1","UV_PYTHON":python,"UV_PYTHON_DOWNLOADS":"never"}
   online=uv_command(cfg,p2a.strings(p2a.mapping(cfg["commands"])["online_sync_args"]),python,cache)
   if not faults:
    side_receipt["online_sync"]=bounded.run_sandboxed(online,repo,side_root,env,cfg,network_denied=False)
    if base.command_failed(side_receipt["online_sync"]):faults.append(f"{label}:online_sync_failed")
   side_receipt["online_environment"]=inspect_environment(repo/".venv",cfg,side) if not faults else {}
   if not faults:faults.extend(f"{label}:online:{e}" for e in side_receipt["online_environment"]["faults"])
   if (repo/".venv").exists():shutil.rmtree(repo/".venv")
   offline=uv_command(cfg,p2a.strings(p2a.mapping(cfg["commands"])["offline_replay_args"]),python,cache)
   if not faults:
    side_receipt["offline_replay"]=bounded.run_sandboxed(offline,repo,side_root,env,cfg,network_denied=True)
    if base.command_failed(side_receipt["offline_replay"]):faults.append(f"{label}:offline_replay_failed")
   side_receipt["offline_environment"]=inspect_environment(repo/".venv",cfg,side) if not faults else {}
   if not faults:faults.extend(f"{label}:offline:{e}" for e in side_receipt["offline_environment"]["faults"])
   if not faults and side_receipt["online_environment"].get("distributions")!=side_receipt["offline_environment"].get("distributions"):faults.append(f"{label}:online_offline_environment_mismatch")
   source_after=base.tree_identity(repo,excluded_roots={".venv"}) if repo.exists() else {};side_receipt["repository_source_before"]=source_before;side_receipt["repository_source_after"]=source_after
   if source_before!=source_after:faults.append(f"{label}:repository_source_mutated_outside_venv")
   side_receipt["cache"]=tree_identity(cache);retained=int(side_receipt["cache"]["bytes"])
   if retained>int(limits["maximum_retained_bytes_per_side"]):faults.append(f"{label}:retained_store_size_boundary_hit")
   store=p2a.resolve(str(side["retained_store"]));
   if not faults:store.parent.mkdir(parents=True,exist_ok=True);os.replace(cache,store)
   side_receipt["retained_store"]=tree_identity(store) if store.exists() else {}
  total=sum(int(p2a.mapping(p2a.mapping(receipts["sides"]).get(label)).get("retained_store",{}).get("bytes") or 0) for label in ("parent","target"));receipts["retained_total_bytes"]=total
  if total>int(limits["maximum_total_retained_bytes"]):faults.append("total_retained_size_boundary_hit")
  if shutil.disk_usage(ROOT).free<int(limits["minimum_free_bytes_after_execution"]):faults.append("free_space_reserve_postflight_boundary_hit")
 receipts["free_bytes_after"]=shutil.disk_usage(ROOT).free;return finish(before,faults,receipts,True)
def uv_command(cfg:dict[str,Any],args:list[str],python:str,cache:Path)->list[str]:return [str(p2a.resolve(str(p2a.mapping(p2a.mapping(cfg["tools"])["uv"])["path"]))),*args,"--python",python,"--cache-dir",str(cache)]
def inspect_environment(venv:Path,cfg:dict[str,Any],side:dict[str,Any])->dict[str,Any]:
 faults=[];rows=[];site=venv/"lib/python3.12/site-packages";parser=email.parser.Parser()
 for meta in sorted(site.glob("*.dist-info/METADATA")) if site.is_dir() else []:
  try:m=parser.parsestr(meta.read_text(errors="replace"));name=str(m.get("Name") or "").lower().replace("_","-");version=str(m.get("Version") or "")
  except OSError:name="";version=""
  if name and version:rows.append({"name":name,"version":version,"metadata_sha256":p2a.sha256_file(meta)})
 locked={(r["name"].lower().replace("_","-"),str(r["version"])) for r in lock_packages(p2a.resolve(str(side["archive"])),str(side["archive_root"])) if r.get("version")}
 for row in rows:
  if (row["name"],row["version"]) not in locked:faults.append(f"installed_distribution_not_locked:{row['name']}@{row['version']}")
 names={r["name"] for r in rows}
 for required in ("httpx","pydantic","websockets"):
  if required not in names:faults.append(f"required_runtime_dependency_missing:{required}")
 if "scrapebadger" in names:faults.append("project_was_installed")
 return {"distribution_count":len(rows),"distributions":rows,"identity_sha256":base.digest_json(rows),"faults":sorted(set(faults))}
def lock_packages(archive:Path,root:str)->list[dict[str,Any]]:
 with tarfile.open(archive,"r:gz") as h:m=h.getmember(root+"/uv.lock");e=h.extractfile(m);v=tomllib.loads((e.read() if e else b"").decode())
 return sorted([{"name":x.get("name"),"version":x.get("version"),"source":x.get("source"),"sdist":x.get("sdist"),"wheels":x.get("wheels",[])} for x in v.get("package",[])],key=lambda r:(str(r["name"]),str(r["version"])))
def member_identity(archive:Path,root:str,relative:str)->dict[str,Any]:
 with tarfile.open(archive,"r:gz") as h:m=h.getmember(root+"/"+relative);e=h.extractfile(m);b=e.read() if e else b"";return {"path":relative,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
def tree_identity(root:Path)->dict[str,Any]:
 files=[p for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()] if root.exists() else [];rows=[{"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":p2a.sha256_file(p)} for p in files];return {"path":p2a.rel(root),"file_count":len(files),"bytes":sum(p.stat().st_size for p in files),"identity_sha256":base.digest_json(rows)}
def finish(before:dict[str,Any],faults:list[str],receipts:dict[str,Any],executed:bool)->dict[str,Any]:
 sides=p2a.mapping(receipts.get("sides"));online=sum(int("online_sync" in p2a.mapping(s)) for s in sides.values());offline=sum(int("offline_replay" in p2a.mapping(s)) for s in sides.values());return {**before,"created_utc":p2a.now(),"trigger_state":"GREEN" if executed and not faults else "RED","state":"TASK_14_SEPARATE_PARENT_TARGET_UV_CLOSURES_ACQUIRED_AND_OFFLINE_REPLAY_QUALIFIED" if executed and not faults else "TASK_14_DUAL_DEPENDENCY_CANARY_FAILED","faults":sorted(set(faults)),"execution_performed":executed,"network_enabled_dependency_syncs":online,"network_denied_offline_replays":offline,"receipts":receipts}
def summary(r:dict[str,Any])->dict[str,Any]:return {k:r.get(k) for k in ("trigger_state","state","execution_performed","network_enabled_dependency_syncs","network_denied_offline_replays","source_build_executions","project_installations","repository_runner_executions","parent_target_or_evaluator_executions","candidate_or_control_calls","external_reference_calls","faults")}
if __name__=="__main__":raise SystemExit(main())
