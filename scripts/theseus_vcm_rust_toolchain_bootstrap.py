#!/usr/bin/env python3
"""Materialize exact Rust 1.97.1 for frozen VCM task 36 only."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
POLICY="project_theseus_vcm_rust_toolchain_bootstrap_v1"; DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_rust_toolchain_bootstrap.json"

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG)); ap.add_argument("--out",default=""); ap.add_argument("--acquire",action="store_true"); a=ap.parse_args()
    path=p2a.resolve(a.config); cfg=p2a.read_json(path); report=acquire(cfg,path) if a.acquire else preflight(cfg,path); p2a.write_json(p2a.resolve(a.out or cfg["report"]),report); print(json.dumps(summary(report),indent=2,sort_keys=True)); return 0 if report["trigger_state"] in {"GREEN","PAUSED"} else 2

def preflight(cfg:dict[str,Any],path:Path)->dict[str,Any]:
    faults=[]
    if cfg.get("policy")!=POLICY or cfg.get("state")!="PROSPECTIVE_EXACT_RUST_1_97_1_FOR_VCM_TASK_36_ONLY": faults.append("policy_or_state_invalid")
    owner=p2a.resolve(str(cfg.get("owner") or ""))
    if owner!=Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner)!=cfg.get("owner_sha256"): faults.append("owner_binding_invalid")
    rustup=p2a.resolve(str(p2a.mapping(cfg.get("rustup")) .get("path") or ""))
    if not rustup.is_file() or p2a.sha256_file(rustup)!=p2a.mapping(cfg.get("rustup")).get("sha256"): faults.append("rustup_binding_invalid")
    manifest=p2a.mapping(cfg.get("official_manifest")); comps=p2a.mapping(manifest.get("minimal_components"))
    if manifest.get("sha256")!="03569b1886ceb5c05276b50c8431ab111de944cd6140fe1fa7d821dd8e0f29cf" or set(comps)!={"rustc","cargo","rust-std"} or any(len(str(p2a.mapping(v).get("xz_sha256") or ""))!=64 for v in comps.values()): faults.append("official_manifest_binding_invalid")
    reports={}
    for name,raw in p2a.mapping(cfg.get("reports")).items():
        b=p2a.mapping(raw); p=p2a.resolve(str(b.get("path") or ""))
        if not p.is_file() or p2a.sha256_file(p)!=b.get("sha256"): faults.append(f"report_binding_invalid:{name}"); reports[name]={}
        else: reports[name]=p2a.read_json(p)
    row=next((r for r in p2a.dicts(reports.get("compatibility_v2",{}).get("rows")) if r.get("index")==36),{})
    if row.get("state")!="INCOMPATIBLE_DECLARED_REQUIREMENTS" or p2a.dicts(row.get("requirements"))[0].get("expression")!="1.97": faults.append("task36_requirement_invalid")
    if reports.get("task7_closure_audit",{}).get("trigger_state")!="GREEN": faults.append("predecessor_boundary_invalid")
    home=p2a.resolve(str(cfg.get("rustup_home") or "")); expected=(ROOT/"runtime/vcm_evaluator/toolchains/rustup-task36").resolve()
    if home!=expected: faults.append("rustup_home_invalid")
    allowed={"exact_official_minimal_toolchain_acquisition_authorized","exact_version_probe_authorized","isolated_toolchain_retention_authorized"}
    for k,v in p2a.mapping(cfg.get("authority")).items():
        if v is not (k in allowed): faults.append(f"authority_invalid:{k}")
    return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"RED" if faults else "PAUSED","state":"CONTRACT_INVALID" if faults else "READY_FOR_EXACT_RUST_1_97_1_ACQUISITION","faults":sorted(set(faults)),"config":identity(path),"acquisition_executed":False,"network_metadata_requests_before_contract":2,"toolchain_acquisition_executions":0,"cargo_dependency_fetches":0,"repository_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0,"maximum_inference":cfg.get("maximum_inference")}

def acquire(cfg:dict[str,Any],path:Path)->dict[str,Any]:
    before=preflight(cfg,path)
    if before["trigger_state"]=="RED": return before
    home=p2a.resolve(str(cfg["rustup_home"])); toolchain=home/"toolchains/1.97.1-aarch64-apple-darwin"
    if toolchain.exists(): return finish(before,inspect(toolchain,cfg),[],False)
    reserve=int(cfg["minimum_free_bytes_after_acquisition"]); free_before=shutil.disk_usage(ROOT).free
    if free_before<reserve: return finish(before,{"free_bytes_before":free_before},["free_space_reserve_preflight_boundary_hit"],False)
    home.mkdir(parents=True,exist_ok=True); rustup=str(p2a.resolve(str(p2a.mapping(cfg["rustup"])["path"])))
    env={"HOME":"/private/tmp","PATH":"/usr/bin:/bin","RUSTUP_HOME":str(home),"RUSTUP_DIST_SERVER":"https://static.rust-lang.org","RUSTUP_UPDATE_ROOT":"https://static.rust-lang.org/rustup","NO_COLOR":"1"}
    command=[rustup,"toolchain","install","1.97.1-aarch64-apple-darwin","--profile","minimal","--no-self-update"]
    done=subprocess.run(command,env=env,capture_output=True,check=False,timeout=int(cfg.get("wall_seconds") or 1200))
    faults=[]
    if done.returncode!=0: faults.append("rustup_acquisition_failed")
    receipt={"command":command,"returncode":done.returncode,"stdout_bytes":len(done.stdout),"stderr_bytes":len(done.stderr),"stdout_sha256":hashlib.sha256(done.stdout).hexdigest(),"stderr_sha256":hashlib.sha256(done.stderr).hexdigest(),"stderr_tail":done.stderr.decode("utf-8","replace")[-2000:],"free_bytes_before":free_before,"free_bytes_after":shutil.disk_usage(ROOT).free}
    if receipt["free_bytes_after"]<reserve: faults.append("free_space_reserve_postflight_boundary_hit")
    receipt["toolchain"]=inspect(toolchain,cfg) if toolchain.is_dir() else {}
    faults.extend(receipt["toolchain"].get("faults",[])); return finish(before,receipt,faults,True)

def inspect(root:Path,cfg:dict[str,Any])->dict[str,Any]:
    faults=[]; bin=root/"bin"; rustc=bin/"rustc"; cargo=bin/"cargo"
    probes={}
    for name,path,expected in (("rustc",rustc,"release: 1.97.1"),("cargo",cargo,"cargo 1.97.1")):
        if not path.is_file(): faults.append(f"{name}_absent"); continue
        done=subprocess.run([str(path),"--version","--verbose"],env={"HOME":"/private/tmp","PATH":"/usr/bin:/bin","NO_COLOR":"1"},capture_output=True,check=False,timeout=30); out=done.stdout.decode("utf-8","replace")
        probes[name]={"returncode":done.returncode,"stdout":out.strip(),"stdout_sha256":hashlib.sha256(done.stdout).hexdigest(),"binary_sha256":p2a.sha256_file(path)}
        if done.returncode!=0 or expected not in out or "host: aarch64-apple-darwin" not in out: faults.append(f"{name}_version_probe_invalid")
    files=[p for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()]
    return {"path":p2a.rel(root),"probes":probes,"file_count":len(files),"bytes":sum(p.stat().st_size for p in files),"faults":sorted(set(faults))}

def finish(before:dict[str,Any],receipt:dict[str,Any],faults:list[str],executed:bool)->dict[str,Any]:
    return {**before,"created_utc":p2a.now(),"trigger_state":"GREEN" if not faults else "RED","state":"EXACT_RUST_1_97_1_MINIMAL_TOOLCHAIN_MATERIALIZED_AND_VERSION_QUALIFIED" if not faults else "RUST_1_97_1_ACQUISITION_FAILED","faults":sorted(set(faults)),"acquisition_executed":executed,"toolchain_acquisition_executions":int(executed),"receipt":receipt}
def identity(p:Path)->dict[str,str]: return {"path":p2a.rel(p),"sha256":p2a.sha256_file(p) if p.is_file() else ""}
def summary(r:dict[str,Any])->dict[str,Any]: return {k:r.get(k) for k in ("trigger_state","state","acquisition_executed","toolchain_acquisition_executions","cargo_dependency_fetches","repository_executions","candidate_or_control_calls","external_reference_calls","faults")}
if __name__=="__main__": raise SystemExit(main())
