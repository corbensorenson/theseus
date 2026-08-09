#!/usr/bin/env python3
"""Role-separated rederivation of the call-free VCM v4 chunk repair."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_k3_route_preflight as v3  # noqa: E402
import theseus_vcm_v4_chunked_route_preflight as owner  # noqa: E402

POLICY = "project_theseus_vcm_v4_chunked_route_preflight_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_v4_chunked_route_preflight.json"

def audit(path: Path = DEFAULT_CONFIG):
    cfg=p2a.read_json(path); actual=p2a.read_json(p2a.resolve(str(cfg.get("report")))); packets=p2a.read_json(p2a.resolve(str(cfg.get("packets_out"))))
    expected, expected_packets=owner.build(path, token_counter=v3.exact_token_counter(p2a.read_json(p2a.resolve(str(cfg.get("v3_config"))))))
    faults=[]
    for key in ("row_count","route_count","packet_count","source_file_count","source_chunk_count","reconstructed_source_file_count","vcm_flat_physically_addressable_matched_pair_count","rows"):
        if actual.get(key)!=expected.get(key): faults.append(f"producer_mismatch:{key}")
    if packets.get("rows")!=expected_packets.get("rows"): faults.append("packet_manifest_mismatch")
    if actual.get("local_model_calls")!=0 or actual.get("hidden_evaluator_calls")!=0 or actual.get("external_reference_calls")!=0: faults.append("call_custody_invalid")
    return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if not faults else "RED","state":"VCM_V4_CHUNKED_ROUTE_PREFLIGHT_ROLE_SEPARATELY_REDERIVED" if not faults else "VCM_V4_CHUNKED_ROUTE_PREFLIGHT_AUDIT_FAILED","faults":sorted(set(faults)),"audited_row_count":expected.get("row_count"),"audited_packet_count":expected.get("packet_count"),"audited_source_chunk_count":expected.get("source_chunk_count"),"audited_matched_pair_count":expected.get("vcm_flat_physically_addressable_matched_pair_count"),"local_model_calls":0,"hidden_evaluator_calls":0,"external_reference_calls":0,"maximum_inference":"A GREEN audit establishes only call-free sub-file reconstruction, route packet identity, matching, and physical addressability for the v4 instrument. It is not model or VCM utility evidence."}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG)); args=parser.parse_args(); path=p2a.resolve(args.config); cfg=p2a.read_json(path); report=audit(path); p2a.write_json(p2a.resolve(str(cfg.get("audit_report"))),report); print(json.dumps(report,indent=2,sort_keys=True)); return 0 if report["trigger_state"]=="GREEN" else 2
if __name__=="__main__": raise SystemExit(main())
