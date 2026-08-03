#!/usr/bin/env python3
"""Independently rederive task 30's retained Cargo dependency closure."""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys,tarfile,tomllib
from pathlib import Path,PurePosixPath
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_dependency_prefetch_canary as base  # noqa:E402
POLICY="project_theseus_vcm_task30_dependency_closure_audit_v1";DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_task30_dependency_closure_audit.json"
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));ap.add_argument("--out",default="");a=ap.parse_args();p=p2a.resolve(a.config);r=audit(p);p2a.write_json(p2a.resolve(a.out or p2a.read_json(p)["report"]),r);print(json.dumps(summary(r),indent=2,sort_keys=True));return 0 if r["trigger_state"]=="GREEN" else 2
def audit(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg=p2a.read_json(path);faults=[]
 if cfg.get("policy")!=POLICY:faults.append("policy_invalid")
 owner=p2a.resolve(str(cfg.get("owner") or ""))
 if owner!=Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner)!=cfg.get("owner_sha256"):faults.append("owner_binding_invalid")
 artifacts={};paths={}
 for name,raw in p2a.mapping(cfg.get("artifacts")).items():
  b=p2a.mapping(raw);p=p2a.resolve(str(b.get("path") or ""));paths[name]=p
  if not p.is_file() or p2a.sha256_file(p)!=b.get("sha256"):faults.append(f"artifact_binding_invalid:{name}");artifacts[name]={}
  else:artifacts[name]=p2a.read_json(p)
 canary=artifacts.get("canary_report",{});ccfg=artifacts.get("canary_config",{});sandbox=artifacts.get("trusted_build_canaries",{})
 if canary.get("trigger_state")!="GREEN" or canary.get("state")!="TASK_30_EXACT_CARGO_DEPENDENCY_CACHE_ACQUIRED_AND_OFFLINE_REPLAY_QUALIFIED":faults.append("canary_not_green")
 if ccfg.get("policy")!="project_theseus_vcm_task30_dependency_canary_v1":faults.append("canary_config_invalid")
 if sandbox.get("trigger_state")!="GREEN" or "network_denial_and_write_confinement" not in p2a.strings(sandbox.get("qualified_scopes")):faults.append("trusted_sandbox_invalid")
 for k,v in p2a.mapping(cfg.get("authority")).items():
  if v is not (k=="static_audit_authorized"):faults.append(f"authority_invalid:{k}")
 receipts=p2a.mapping(canary.get("receipts"));online=p2a.mapping(receipts.get("online_fetch"));offline=p2a.mapping(receipts.get("offline_replay"));commands=p2a.mapping(ccfg.get("commands"));online_args=p2a.strings(commands.get("online_fetch_args"));offline_args=p2a.strings(commands.get("offline_replay_args"))
 command_checks={"online_returncode_zero":online.get("returncode")==0,"offline_returncode_zero":offline.get("returncode")==0,"online_boundary_clear":online.get("boundary_hit") is False,"offline_boundary_clear":offline.get("boundary_hit") is False,"online_network_phase":online.get("network_denied") is False,"offline_deny_network_phase":offline.get("network_denied") is True,"online_command_exact":p2a.strings(online.get("command"))[1:]==online_args,"offline_command_exact":p2a.strings(offline.get("command"))[1:]==offline_args};faults.extend(f"command_check_failed:{k}" for k,v in command_checks.items() if not v)
 checks=lock_checksums(ccfg);check_identity=hashlib.sha256(json.dumps(checks,sort_keys=True,separators=(",",":")).encode()).hexdigest();store=p2a.resolve(str(cfg.get("retained_store") or ""));store_receipt=inspect_store(store,checks);reported=p2a.mapping(receipts.get("retained_store"))
 store_checks={"checksum_denominator_exact":len(checks)==50 and check_identity==p2a.mapping(ccfg.get("task")).get("expected_checksum_identity_sha256"),"all_crates_matched":store_receipt.get("matched_checksum_count")==store_receipt.get("expected_checksum_count")==50,"no_missing_crates":store_receipt.get("missing_packages")==[],"bytes_match":store_receipt.get("bytes")==reported.get("bytes"),"files_match":store_receipt.get("file_count")==reported.get("file_count"),"identity_matches":store_receipt.get("files_identity_sha256")==reported.get("files_identity_sha256"),"crate_receipts_match":store_receipt.get("crate_receipts_sha256")==reported.get("crate_receipts_sha256")};faults.extend(f"store_check_failed:{k}" for k,v in store_checks.items() if not v)
 target=p2a.mapping(p2a.mapping(ccfg.get("archives")).get("target"));source=archive_tree_identity(p2a.resolve(str(target.get("path") or "")),str(target.get("archive_root") or ""));before=p2a.mapping(receipts.get("repository_source_before"));after=p2a.mapping(receipts.get("repository_source_after"));source_checks={"before_after_identical":before==after,"archive_matches_before":source==before,"archive_matches_after":source==after};faults.extend(f"source_check_failed:{k}" for k,v in source_checks.items() if not v)
 online_cache=p2a.mapping(receipts.get("online_cache"));offline_cache=p2a.mapping(receipts.get("offline_cache"));fields=("expected_checksum_count","matched_checksum_count","missing_packages","crate_receipts_sha256","file_count","bytes","files_identity_sha256");phase_checks={"online_faults_empty":online_cache.get("faults")==[],"offline_faults_empty":offline_cache.get("faults")==[],"online_offline_cache_exact":{k:online_cache.get(k) for k in fields}=={k:offline_cache.get(k) for k in fields}};faults.extend(f"phase_check_failed:{k}" for k,v in phase_checks.items() if not v)
 limits=p2a.mapping(ccfg.get("limits"));reserve=int(limits.get("minimum_free_bytes_after_execution") or 0);current=shutil.disk_usage(ROOT).free;storage_checks={"run_before_above_reserve":int(receipts.get("free_bytes_before") or 0)>=reserve,"run_after_above_reserve":int(receipts.get("free_bytes_after") or 0)>=reserve,"current_above_reserve":current>=reserve,"store_below_ceiling":int(store_receipt.get("bytes") or 0)<=int(limits.get("maximum_retained_bytes") or 0)};faults.extend(f"storage_check_failed:{k}" for k,v in storage_checks.items() if not v)
 zero={k:canary.get(k) for k in ("dependency_installations","repository_build_executions","repository_runner_executions","parent_target_or_evaluator_executions","candidate_or_control_calls","external_reference_calls")}
 if any(v!=0 for v in zero.values()):faults.append("downstream_zero_counter_invalid")
 obs={"task_index":30,"lock_checksum_package_count":len(checks),"matched_checksum_count":store_receipt.get("matched_checksum_count"),"retained_store_bytes":store_receipt.get("bytes"),"retained_store_file_count":store_receipt.get("file_count"),"source_file_count":source.get("file_count"),"source_bytes":source.get("bytes"),"current_free_bytes":current,**{k:0 for k in zero}}
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if not faults else "RED","state":"TASK_30_DEPENDENCY_CLOSURE_INDEPENDENTLY_REDERIVED" if not faults else "TASK_30_DEPENDENCY_CLOSURE_AUDIT_FAILED","faults":sorted(set(faults)),"config":base.identity(path),"artifacts":{n:base.identity(p) for n,p in paths.items()},"retained_store":p2a.rel(store),"command_checks":command_checks,"store_checks":store_checks,"source_checks":source_checks,"phase_checks":phase_checks,"storage_checks":storage_checks,"rederived_store":store_receipt,"rederived_source":source,"observations":obs,"static_audit_only":True,"network_or_dependency_execution_performed":False,**{k:0 for k in zero},"maximum_inference":cfg.get("maximum_inference")}
def lock_checksums(cfg:dict[str,Any])->list[dict[str,str]]:
 target=p2a.mapping(p2a.mapping(cfg["archives"])["target"]);archive=p2a.resolve(str(target["path"]));root=str(target["archive_root"]);lock_path=str(p2a.mapping(p2a.mapping(cfg["task"])["required_files"])["lock"]["path"])
 with tarfile.open(archive,"r:gz") as h:m=h.getmember(f"{root}/{lock_path}");e=h.extractfile(m);lock=tomllib.loads((e.read() if e else b"").decode())
 return sorted([{"name":str(x["name"]),"version":str(x["version"]),"checksum":str(x["checksum"])} for x in lock.get("package",[]) if x.get("checksum")],key=lambda x:(x["name"],x["version"]))
def inspect_store(home:Path,expected:list[dict[str,str]])->dict[str,Any]:
 found=[];missing=[]
 for row in expected:
  matches=list((home/"registry/cache").glob(f"*/{row['name']}-{row['version']}.crate")) if (home/"registry/cache").is_dir() else [];good=[p for p in matches if hashlib.sha256(p.read_bytes()).hexdigest()==row["checksum"]]
  if len(good)==1:found.append({**row,"path":good[0].relative_to(home).as_posix(),"bytes":good[0].stat().st_size})
  else:missing.append(f"{row['name']}@{row['version']}")
 files=[p for p in sorted(home.rglob("*")) if p.is_file() and not p.is_symlink()] if home.exists() else [];rows=[{"path":p.relative_to(home).as_posix(),"bytes":p.stat().st_size,"sha256":p2a.sha256_file(p)} for p in files]
 return {"expected_checksum_count":len(expected),"matched_checksum_count":len(found),"missing_packages":missing,"crate_receipts_sha256":base.digest_json(found),"file_count":len(files),"bytes":sum(p.stat().st_size for p in files),"files_identity_sha256":base.digest_json(rows)}
def archive_tree_identity(archive:Path,root:str)->dict[str,Any]:
 rows=[]
 with tarfile.open(archive,"r:gz") as h:
  for m in h.getmembers():
   if m.isfile() and m.name.startswith(root+"/"):
    rel=m.name[len(root)+1:];e=h.extractfile(m);b=e.read() if e else b"";rows.append({"path":rel,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()})
 rows.sort(key=lambda r:PurePosixPath(r["path"]));return {"file_count":len(rows),"bytes":sum(r["bytes"] for r in rows),"identity_sha256":base.digest_json(rows)}
def summary(r:dict[str,Any])->dict[str,Any]:return {k:r.get(k) for k in ("trigger_state","state","observations","static_audit_only","network_or_dependency_execution_performed","repository_build_executions","repository_runner_executions","parent_target_or_evaluator_executions","candidate_or_control_calls","external_reference_calls","faults")}
if __name__=="__main__":raise SystemExit(main())
