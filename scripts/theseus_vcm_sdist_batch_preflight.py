#!/usr/bin/env python3
"""Acquire and statically classify a manifest-driven batch of sdist-only closures."""
from __future__ import annotations
import argparse,ast,hashlib,json,shutil,subprocess,sys,tarfile,tempfile,tomllib
from pathlib import Path,PurePosixPath
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
POLICY="project_theseus_vcm_sdist_batch_preflight_v1";STATE="PROSPECTIVE_TASK13_TRANSITIVE_SDIST_BATCH_RETENTION_AND_POETRY_REPAIR_V2";DEFAULT_CONFIG=ROOT/"configs/theseus_vcm_sdist_batch_preflight.json"
DANGEROUS={"subprocess","popen","system","socket","urlopen","requests","curl","wget","ctypes","eval","exec"}
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));ap.add_argument("--out",default="");ap.add_argument("--execute",action="store_true");a=ap.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=execute(path) if a.execute else preflight_report(path);p2a.write_json(p2a.resolve(a.out or cfg["report"]),r);print(json.dumps({k:r.get(k) for k in ("trigger_state","state","faults","execution_performed","row_count","eligible_row_count","inconclusive_row_count","source_build_executions","candidate_or_control_calls","external_reference_calls")},indent=2,sort_keys=True));return 0 if r["trigger_state"] in {"GREEN","PAUSED"} else 2
def preflight(path:Path=DEFAULT_CONFIG)->tuple[dict[str,Any],dict[str,Any],list[str]]:
 cfg=p2a.read_json(path);faults=[]
 if cfg.get("policy")!=POLICY or cfg.get("state")!=STATE:faults.append("policy_or_state_invalid")
 for key,expected in (("owner",Path(__file__).resolve()),("audit_owner",ROOT/"scripts/theseus_vcm_sdist_batch_preflight_audit.py")):
  owner=p2a.resolve(str(cfg.get(key) or ""))
  if owner!=expected.resolve() or not owner.is_file() or p2a.sha256_file(owner)!=cfg.get(f"{key}_sha256"):faults.append(f"{key}_binding_invalid")
 sources={}
 for name,raw in p2a.mapping(cfg.get("sources")).items():
  b=p2a.mapping(raw);p=p2a.resolve(str(b.get("path") or ""))
  if not p.is_file() or p2a.sha256_file(p)!=b.get("sha256"):faults.append(f"source_binding_invalid:{name}");sources[name]={}
  else:sources[name]=p2a.read_json(p)
 producer=sources.get("immutable_resolution_v4",{});audit=sources.get("immutable_resolution_audit_v4",{});row13=next((r for r in p2a.dicts(producer.get("rows")) if r.get("index")==13),{});stderr=str(p2a.mapping(row13.get("receipt")).get("stderr") or "")
 if producer.get("trigger_state")!="GREEN" or producer.get("qualified_task_count")!=5 or row13.get("disposition")!="INCONCLUSIVE_EXPERIMENT_DEPENDENCY_RESOLUTION":faults.append("v4_producer_boundary_invalid")
 if audit.get("trigger_state")!="GREEN" or audit.get("qualified_task_count")!=5:faults.append("v4_audit_boundary_invalid")
 prior=sources.get("batch_v1",{});prior_audit=sources.get("batch_v1_audit",{})
 if prior.get("trigger_state")!="GREEN" or prior.get("eligible_row_count")!=1 or prior.get("inconclusive_row_count")!=2:faults.append("batch_v1_boundary_invalid")
 expected_prior_faults={"bluetooth-data-tools:retained_sdist_missing","habluetooth:retained_sdist_missing","row_rederivation_invalid:bluetooth-data-tools","row_rederivation_invalid:habluetooth"}
 if prior_audit.get("trigger_state")!="RED" or set(p2a.strings(prior_audit.get("faults")))!=expected_prior_faults:faults.append("batch_v1_audit_wall_invalid")
 rows=p2a.dicts(cfg.get("packages"));expected=[("habluetooth","6.26.5"),("bluetooth-data-tools","1.29.21"),("PyRIC","0.1.6.3")]
 if [(r.get("name"),r.get("version")) for r in rows]!=expected:faults.append("batch_denominator_invalid")
 for row in rows:
  if str(row.get("name") or "").lower().replace("-","") not in stderr.lower().replace("-",""):faults.append(f"package_not_bound_to_v4_stderr:{row.get('name')}")
  if not str(row.get("url") or "").startswith("https://files.pythonhosted.org/") or len(str(row.get("sha256") or ""))!=64:faults.append(f"package_identity_invalid:{row.get('name')}")
 curl=Path(str(p2a.mapping(cfg.get("curl")).get("path") or ""))
 if curl!=Path("/usr/bin/curl") or not curl.is_file() or p2a.sha256_file(curl)!=p2a.mapping(cfg.get("curl")).get("sha256"):faults.append("curl_binding_invalid")
 store=p2a.resolve(str(cfg.get("store") or ""))
 if store!=(ROOT/"runtime/vcm_evaluator/dependency_store/sdist-batch/task-13").resolve():faults.append("store_binding_invalid")
 allowed={"exact_sdist_download_authorized","static_archive_inspection_authorized","content_addressed_sdist_retention_authorized"}
 for key,value in p2a.mapping(cfg.get("authority")).items():
  if value is not (key in allowed):faults.append(f"authority_invalid:{key}")
 return cfg,{"sources":sources,"store":store},sorted(set(faults))
def preflight_report(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg,_,faults=preflight(path);return finish(cfg,path,faults,[],False)
def execute(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg,bound,faults=preflight(path)
 if faults:return finish(cfg,path,faults,[],False)
 limits=p2a.mapping(cfg["limits"]);store:Path=bound["store"]
 if shutil.disk_usage(ROOT).free-int(limits["maximum_total_download_bytes"])<int(limits["minimum_free_bytes_after_execution"]):return finish(cfg,path,["free_space_reserve_preflight_boundary_hit"],[],False)
 results=[]
 with tempfile.TemporaryDirectory(prefix="theseus-vcm-sdist-batch-",dir="/private/tmp") as raw:
  temp=Path(raw)
  for row in p2a.dicts(cfg["packages"]):
   downloaded=temp/str(row["filename"]);target=store/str(row["filename"]);reused=target.is_file() and p2a.sha256_file(target)==row["sha256"];cmd=[]
   if reused:shutil.copyfile(target,downloaded);done=subprocess.CompletedProcess([],0,b"",b"")
   else:
    cmd=["/usr/bin/curl","--fail","--location","--silent","--show-error","--max-time",str(limits["network_timeout_seconds"]),"--max-filesize",str(limits["maximum_single_download_bytes"]),"--output",str(downloaded),str(row["url"])];done=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
   payload=downloaded.read_bytes() if downloaded.is_file() else b"";row_faults=[]
   if done.returncode!=0:row_faults.append("download_transport_failed")
   if hashlib.sha256(payload).hexdigest()!=row["sha256"]:row_faults.append("download_sha256_mismatch")
   inspection,inspect_faults=inspect_archive(downloaded,limits,p2a.strings(cfg["allowed_build_backend_prefixes"])) if not row_faults else ({},[]);row_faults.extend(inspect_faults);eligible=not row_faults and inspection.get("eligible") is True
   if hashlib.sha256(payload).hexdigest()==row["sha256"]:
    store.mkdir(parents=True,exist_ok=True)
    if not target.exists():shutil.copyfile(downloaded,target)
   results.append({"name":row["name"],"version":row["version"],"download":{"command":cmd,"returncode":done.returncode,"bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest(),"stderr":done.stderr.decode("utf-8","replace"),"stderr_sha256":hashlib.sha256(done.stderr).hexdigest(),"reused_from_v1_store":reused},"inspection":inspection,"retained_sdist":{"path":p2a.rel(target),"bytes":target.stat().st_size,"sha256":p2a.sha256_file(target)} if target.is_file() else {},"faults":sorted(set(row_faults)),"disposition":"ELIGIBLE_FOR_NETWORK_DENIED_BATCH_BUILD" if eligible else "INCONCLUSIVE_IMPLEMENTATION_UNTRUSTED_BUILD_PREFLIGHT"})
 return finish(cfg,path,faults,results,True)
def inspect_archive(path:Path,limits:dict[str,Any],allowed:list[str])->tuple[dict[str,Any],list[str]]:
 faults=[];rows=[];roots=set();license_paths=[];pyproject=b"";setup=b""
 try:
  with tarfile.open(path,"r:gz") as h:
   for member in h.getmembers():
    pure=PurePosixPath(member.name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:faults.append("unsafe_member_path");continue
    roots.add(pure.parts[0])
    if not(member.isfile() or member.isdir()):faults.append(f"unsupported_member:{member.name}");continue
    if not member.isfile():continue
    if member.size>int(limits["maximum_single_member_bytes"]):faults.append(f"member_too_large:{member.name}");continue
    e=h.extractfile(member);payload=e.read() if e else b"";rel=PurePosixPath(*pure.parts[1:]).as_posix();rows.append({"path":rel,"bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest()})
    if rel=="pyproject.toml":pyproject=payload
    if rel=="setup.py":setup=payload
    if PurePosixPath(rel).name.lower().startswith(("license","copying")):license_paths.append(rel)
 except (tarfile.TarError,OSError):faults.append("archive_parse_failed")
 if len(roots)!=1:faults.append("archive_root_ambiguous")
 if len(rows)>int(limits["maximum_member_count"]) or sum(r["bytes"] for r in rows)>int(limits["maximum_uncompressed_bytes"]):faults.append("archive_expansion_boundary_hit")
 backend="setuptools.build_meta:__legacy__";requires=["setuptools==80.9.0","wheel==0.45.1"]
 if pyproject:
  try:
   build=p2a.mapping(tomllib.loads(pyproject.decode("utf-8" )).get("build-system"));backend=str(build.get("build-backend") or "");requires=p2a.strings(build.get("requires"))
  except (UnicodeDecodeError,tomllib.TOMLDecodeError):faults.append("pyproject_build_system_parse_failed")
 setup_analysis=analyze_setup(setup)
 if setup_analysis["dangerous_tokens"]:faults.append("setup_py_dangerous_ast_call")
 if not license_paths:faults.append("license_file_missing")
 if not backend or not any(backend.startswith(prefix) for prefix in allowed):faults.append("build_backend_not_allowlisted")
 if not requires:faults.append("build_requirements_missing")
 eligible=not faults
 ordered=sorted(rows,key=lambda r:r["path"])
 return {"eligible":eligible,"archive_root_count":len(roots),"regular_file_count":len(rows),"regular_file_bytes":sum(r["bytes"] for r in rows),"member_receipts_sha256":hashlib.sha256(json.dumps(ordered,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"license_paths":sorted(license_paths),"build_backend":backend,"build_requirements":requires,"setup_py":setup_analysis},sorted(set(faults))
def analyze_setup(payload:bytes)->dict[str,Any]:
 imports=set();calls=set();parse_fault=""
 if payload:
  try:
   tree=ast.parse(payload.decode("utf-8","replace"))
   for node in ast.walk(tree):
    if isinstance(node,ast.Import):imports.update(a.name for a in node.names)
    elif isinstance(node,ast.ImportFrom):imports.add(str(node.module or ""))
    elif isinstance(node,ast.Call):calls.add(call_name(node.func))
  except SyntaxError as e:parse_fault=f"SyntaxError:{e.lineno}"
 names={n.lower() for n in imports|calls};dangerous=sorted(t for t in DANGEROUS if any(n==t or n.startswith(t+".") or n.endswith("."+t) for n in names))
 return {"present":bool(payload),"bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest(),"imports":sorted(imports),"calls":sorted(calls),"dangerous_tokens":dangerous,"parse_fault":parse_fault}
def call_name(node:ast.AST)->str:
 if isinstance(node,ast.Name):return node.id
 if isinstance(node,ast.Attribute):return f"{call_name(node.value)}.{node.attr}".strip(".")
 return ""
def finish(cfg:dict[str,Any],path:Path,faults:list[str],rows:list[dict[str,Any]],execution:bool)->dict[str,Any]:
 eligible=sum(r.get("disposition")=="ELIGIBLE_FOR_NETWORK_DENIED_BATCH_BUILD" for r in rows)
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"RED" if faults else ("GREEN" if execution else "PAUSED"),"state":"TASK13_TRANSITIVE_SDIST_BATCH_STATIC_PREFLIGHT_COMPLETE" if execution and not faults else ("READY_FOR_TASK13_TRANSITIVE_SDIST_BATCH_STATIC_PREFLIGHT" if not faults else "TASK13_TRANSITIVE_SDIST_BATCH_STATIC_PREFLIGHT_INVALID"),"faults":sorted(set(faults)),"config":{"path":p2a.rel(path),"sha256":p2a.sha256_file(path)},"execution_performed":execution,"row_count":len(rows),"eligible_row_count":eligible,"inconclusive_row_count":len(rows)-eligible,"rows":rows,"partial_batch_admission":False,"source_build_executions":0,"package_installations":0,"repository_runner_executions":0,"parent_target_or_evaluator_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0,"teacher_calls":0,"next_authorized_boundary":"seal_network_denied_manifest_driven_batch_build_for_eligible_rows" if rows and eligible==len(rows) else "repair_inconclusive_static_rows","maximum_inference":cfg.get("maximum_inference")}
if __name__=="__main__":raise SystemExit(main())
