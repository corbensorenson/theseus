from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_v4_chunked_route_preflight as owner  # noqa: E402
CONFIG=ROOT/"configs"/"theseus_vcm_v4_chunked_route_preflight.json"

def fake_counter(system: str, prompt: str):
    value=max(1,(len(system.encode())+len(prompt.encode()))//4)
    return {"kind":"injected_exact","exact_tokens":value,"lower_bound_tokens":value}

def test_chunking_is_exact_and_newline_preferring():
    text="alpha\n"+"x"*20+"\nomega\n"
    rows=owner.chunk_text("a.txt",text,12)
    assert "".join(row["text"] for row in rows)==text
    assert all(row["chars"]<=12 for row in rows)
    assert [row["start_char"] for row in rows][1:]==[row["end_char"] for row in rows][:-1]

def test_v4_call_free_build_has_six_matched_pairs_and_no_calls():
    report,packets=owner.build(CONFIG,token_counter=fake_counter)
    assert report["trigger_state"]=="GREEN"
    assert report["row_count"]==6 and report["packet_count"]==36
    assert report["source_file_count"]==report["reconstructed_source_file_count"]
    assert report["vcm_flat_physically_addressable_matched_pair_count"]==6
    assert report["consumed_v3_prompt_identity_count"]>=1
    assert report["local_model_calls"]==0 and report["hidden_evaluator_calls"]==0
    assert len(packets["rows"])==36
    assert all(row["new_host_call_authorized"] is False for row in packets["rows"])

def test_vcm_and_flat_share_exact_chunk_information():
    report,_=owner.build(CONFIG,token_counter=fake_counter)
    for row in report["rows"]:
        arms={arm["route"]:arm for arm in row["arms"]}
        assert arms["governed_vcm"]["context_information_sha256"]==arms["information_matched_flat_direct_context"]["context_information_sha256"]
