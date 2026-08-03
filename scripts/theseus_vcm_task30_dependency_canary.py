#!/usr/bin/env python3
"""Fetch and offline-replay only task 30's exact Cargo dependency lock."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,sys,tarfile,tempfile,tomllib
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_dependency_prefetch_canary as base  # noqa:E402
import theseus_vcm_dependency_prefetch_canary_v2 as bounded  # noqa:E402
POLICY="project_theseus_vcm_task30_dependency_canary_v1";STATE="PROSPECTIVE_TASK_30_EXACT_CARGO_DEPENDENCY_FETCH_AND_OFFLINE_REPLAY";DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_task30_dependency_canary.json"
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));ap.add_argument("--out",default="");ap.add_argument("--execute",action="store_true");a=ap.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=execute(cfg,path) if a.execute else preflight(cfg,path);p2a.write_json(p2a.resolve(a.out or cfg["report"]),r);print(json.dumps(summary(r),indent=2,sort_keys=True));return 0 if r["trigger_state"] in {"GREEN","PAUSED"} else 2
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
 row=next((r for r in p2a.dicts(reports.get("prefetch_plan",{}).get("schedule")) if r.get("schedule_ordinal")==5),{});compat=next((r for r in p2a.dicts(reports.get("compatibility_v3",{}).get("rows")) if r.get("index")==30),{});pair=next((r for r in p2a.dicts(reports.get("pair_coverage",{}).get("rows")) if r.get("index")==30),{})
 if row.get("index")!=30 or row.get("manager")!="cargo" or p2a.mapping(row.get("governing_lock")).get("sha256")!="674d54461401ec1680d22001b7411e505ee026c1d6c0796038ced82765ea23c8":faults.append("schedule_binding_invalid")
 if compat.get("state")!="NO_DECLARED_VERSION_REQUIREMENTS_TOOL_AVAILABLE" or "cargo1_97_1_rustc1_97_1" not in compat.get("compatible_profile_ids",[]):faults.append("compatibility_binding_invalid")
 if pair.get("state")!="IDENTICAL_PARENT_TARGET_DEPENDENCY_IDENTITY" or pair.get("required_distinct_closure_count")!=1:faults.append("pair_coverage_binding_invalid")
 if reports.get("rust_toolchain",{}).get("trigger_state")!="GREEN" or reports.get("task14_closure_audit",{}).get("trigger_state")!="GREEN":faults.append("predecessor_report_invalid")
 task=p2a.mapping(cfg.get("task"))
 if task.get("index")!=30 or task.get("repository")!="kanso-lang/kanso" or task.get("expected_checksum_package_count")!=50:faults.append("task_binding_invalid")
 observed={}
 for label,raw in p2a.mapping(cfg.get("archives")).items():
  b=p2a.mapping(raw);p=p2a.resolve(str(b.get("path") or ""))
  if not p.is_file() or p2a.sha256_file(p)!=b.get("sha256"):faults.append(f"archive_binding_invalid:{label}");continue
  ids,errs=base.archive_lock_identities(p,str(b.get("archive_root") or ""),cfg);faults.extend(f"{label}:{e}" for e in errs);observed[label]=ids
 if observed.get("parent")!=observed.get("target"):faults.append("parent_target_dependency_identity_mismatch")
 checks=lock_checksums(cfg);identity=hashlib.sha256(json.dumps(checks,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 if len(checks)!=50 or identity!=task.get("expected_checksum_identity_sha256"):faults.append("lock_checksum_denominator_invalid")
 for name,raw in p2a.mapping(cfg.get("tools")).items():
  b=p2a.mapping(raw);p=p2a.resolve(str(b.get("path") or ""))
  if not p.is_file() or p2a.sha256_file(p)!=b.get("sha256"):faults.append(f"tool_binding_invalid:{name}")
 commands=p2a.mapping(cfg.get("commands"));online=p2a.strings(commands.get("online_fetch_args"));offline=p2a.strings(commands.get("offline_replay_args"))
 if online!=["fetch","--locked","--config","net.git-fetch-with-cli=false"] or offline!=[*online,"--offline"]:faults.append("command_contract_invalid")
 allowed={"temporary_normalized_archive_extraction_authorized","single_network_dependency_fetch_authorized","network_denied_offline_replay_authorized","content_addressed_cache_retention_authorized"}
 for k,v in p2a.mapping(cfg.get("authority")).items():
  if v is not (k in allowed):faults.append(f"authority_invalid:{k}")
 store=p2a.resolve(str(cfg.get("retained_store") or ""))
 if store.parent!=(ROOT/"runtime/vcm_evaluator/dependency_store/cargo").resolve() or store.name!="task-30":faults.append("retained_store_invalid")
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"RED" if faults else "PAUSED","state":"CONTRACT_INVALID" if faults else "READY_FOR_TASK_30_EXACT_CARGO_DEPENDENCY_CANARY","faults":sorted(set(faults)),"config":base.identity(path),"observed_parent_target_dependency_identities":observed,"lock_checksum_package_count":len(checks),"lock_checksum_identity_sha256":identity,"execution_performed":False,"network_enabled_dependency_fetches":0,"network_denied_offline_replays":0,"dependency_installations":0,"repository_build_executions":0,"repository_runner_executions":0,"parent_target_or_evaluator_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0,"maximum_inference":cfg.get("maximum_inference")}
def execute(cfg:dict[str,Any],path:Path)->dict[str,Any]:
 before=preflight(cfg,path)
 if before["trigger_state"]=="RED":return before
 store=p2a.resolve(str(cfg["retained_store"]));limits=p2a.mapping(cfg["limits"])
 if store.exists():return finish(before,["retained_store_already_exists"],{},False)
 free=shutil.disk_usage(ROOT).free
 if free<int(limits["minimum_free_bytes_after_execution"]):return finish(before,["free_space_reserve_preflight_boundary_hit"],{"free_bytes_before":free},False)
 faults=[];receipts={"free_bytes_before":free}
 with tempfile.TemporaryDirectory(prefix="theseus-vcm-task30-deps-",dir="/private/tmp") as raw:
  work=Path(raw).resolve();repo=work/"repository";cargo_home=work/"cargo-home";home=work/"home";tmp=work/"tmp";cargo_home.mkdir();home.mkdir();tmp.mkdir();target=p2a.mapping(p2a.mapping(cfg["archives"])["target"])
  receipts["repository_extraction"],errs=base.safe_extract_repository(p2a.resolve(str(target["path"])),repo,str(target["archive_root"]));faults.extend(errs);source_before=base.tree_identity(repo,excluded_roots={"target"}) if not faults else {}
  tools=p2a.mapping(cfg["tools"]);env={"HOME":str(home),"TMPDIR":str(tmp),"PATH":"/usr/bin:/bin","CARGO_HOME":str(cargo_home),"RUSTC":str(p2a.resolve(str(p2a.mapping(tools["rustc"])["path"]))),"CI":"1","NO_COLOR":"1","CARGO_TERM_COLOR":"never"}
  online=cargo_command(cfg,p2a.strings(p2a.mapping(cfg["commands"])["online_fetch_args"]))
  if not faults:receipts["online_fetch"]=bounded.run_sandboxed(online,repo,work,env,cfg,network_denied=False);faults.extend(["online_dependency_fetch_failed"] if base.command_failed(receipts["online_fetch"]) else [])
  receipts["online_cache"]=inspect_cache(cargo_home,cfg) if not faults else {}
  if not faults:faults.extend(f"online:{e}" for e in receipts["online_cache"]["faults"])
  offline=cargo_command(cfg,p2a.strings(p2a.mapping(cfg["commands"])["offline_replay_args"]))
  if not faults:receipts["offline_replay"]=bounded.run_sandboxed(offline,repo,work,env,cfg,network_denied=True);faults.extend(["offline_dependency_replay_failed"] if base.command_failed(receipts["offline_replay"]) else [])
  receipts["offline_cache"]=inspect_cache(cargo_home,cfg) if not faults else {};source_after=base.tree_identity(repo,excluded_roots={"target"}) if repo.exists() else {};receipts["repository_source_before"]=source_before;receipts["repository_source_after"]=source_after
  if not faults:faults.extend(f"offline:{e}" for e in receipts["offline_cache"]["faults"])
  if source_before!=source_after:faults.append("repository_source_mutated_outside_target")
  retained=base.directory_bytes(cargo_home)
  if retained>int(limits["maximum_retained_bytes"]):faults.append("retained_store_size_boundary_hit")
  if shutil.disk_usage(ROOT).free-retained<int(limits["minimum_free_bytes_after_execution"]):faults.append("free_space_reserve_postflight_boundary_hit")
  if not faults:store.parent.mkdir(parents=True,exist_ok=True);os.replace(cargo_home,store)
 receipts["retained_store"]=inspect_cache(store,cfg) if store.exists() else {};receipts["free_bytes_after"]=shutil.disk_usage(ROOT).free;return finish(before,faults,receipts,True)
def cargo_command(cfg:dict[str,Any],args:list[str])->list[str]:return [str(p2a.resolve(str(p2a.mapping(p2a.mapping(cfg["tools"])["cargo"])["path"]))),*args]
def lock_checksums(cfg:dict[str,Any])->list[dict[str,str]]:
 target=p2a.mapping(p2a.mapping(cfg["archives"])["target"]);archive=p2a.resolve(str(target["path"]));root=str(target["archive_root"]);lock_path=str(p2a.mapping(p2a.mapping(cfg["task"])["required_files"])["lock"]["path"])
 with tarfile.open(archive,"r:gz") as h:m=h.getmember(f"{root}/{lock_path}");e=h.extractfile(m);lock=tomllib.loads((e.read() if e else b"").decode())
 return sorted([{"name":str(x["name"]),"version":str(x["version"]),"checksum":str(x["checksum"])} for x in lock.get("package",[]) if x.get("checksum")],key=lambda x:(x["name"],x["version"]))
def inspect_cache(home:Path,cfg:dict[str,Any])->dict[str,Any]:
 expected=lock_checksums(cfg);found=[];missing=[]
 for row in expected:
  matches=list((home/"registry/cache").glob(f"*/{row['name']}-{row['version']}.crate")) if (home/"registry/cache").is_dir() else [];good=[p for p in matches if hashlib.sha256(p.read_bytes()).hexdigest()==row["checksum"]]
  if len(good)==1:found.append({**row,"path":good[0].relative_to(home).as_posix(),"bytes":good[0].stat().st_size})
  else:missing.append(f"{row['name']}@{row['version']}")
 files=[p for p in sorted(home.rglob("*")) if p.is_file() and not p.is_symlink()] if home.exists() else []
 return {"path":p2a.rel(home),"expected_checksum_count":len(expected),"matched_checksum_count":len(found),"missing_packages":missing,"crate_receipts_sha256":base.digest_json(found),"file_count":len(files),"bytes":sum(p.stat().st_size for p in files),"files_identity_sha256":base.digest_json([{"path":p.relative_to(home).as_posix(),"bytes":p.stat().st_size,"sha256":p2a.sha256_file(p)} for p in files]),"faults":[] if not missing else ["lock_checksum_crates_missing"]}
def finish(before:dict[str,Any],faults:list[str],receipts:dict[str,Any],executed:bool)->dict[str,Any]:
 return {**before,"created_utc":p2a.now(),"trigger_state":"GREEN" if executed and not faults else "RED","state":"TASK_30_EXACT_CARGO_DEPENDENCY_CACHE_ACQUIRED_AND_OFFLINE_REPLAY_QUALIFIED" if executed and not faults else "TASK_30_DEPENDENCY_CANARY_FAILED","faults":sorted(set(faults)),"execution_performed":executed,"network_enabled_dependency_fetches":int("online_fetch" in receipts),"network_denied_offline_replays":int("offline_replay" in receipts),"receipts":receipts}
def summary(r:dict[str,Any])->dict[str,Any]:return {k:r.get(k) for k in ("trigger_state","state","execution_performed","network_enabled_dependency_fetches","network_denied_offline_replays","dependency_installations","repository_build_executions","repository_runner_executions","parent_target_or_evaluator_executions","candidate_or_control_calls","external_reference_calls","faults")}
if __name__=="__main__":raise SystemExit(main())
