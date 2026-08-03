# Theseus Assistant Runtime

- trigger_state: `GREEN`
- intent: `code`
- session: `p2a_semantic_ir_adequacy_fresh_v5_01_direct_local_model_1`
- lane: `direct_fixed_model_runtime`
- VCM: `None` pages=`None`
- VCM governor: `None` ready=`None` adequacy=`None` mission=`None` deletion=`None`
- code probe: `None` safe=`None` selected_pass=`None` pass_if_any=`None` integrity_mismatches=`None`
- tool evidence: `None` results=`None` exact_solve=`None`
- procedural default route: active=`None` ready=`None` matched=`None` route=`None` scope=`None` guard=`None` learned_claim_allowed=`None`
- VIEA trace: required=`None` complete=`None` records=`None` out=`None`
- VIEA materialized view: ready=`None` records=`None` claims=`None`
- private verifier receipt: ready=`None` state=`None` records=`None`
- latest public: `None` score=`None` tasks=`None` kind=`None`
- answer chars: `487`
- feedback: `completed`
- assistant trace schema: `configs/assistant_trace_schema.json` ready=`True`
- dogfood event written: `False`
- dogfood rows written: `0`
- teacher distillation: `None` allowed=`None` share=`None` runtime_tokens_forbidden=`None`

## Answer

THESEUS_SEMANTIC_IR_V3
SOURCE 8c68e30b6bbcdb12ce11d0dc5a5d47631faace4e9c70c1255232c077352704d7
ALL_OBLIGATIONS O1,O2,O3
CHANGE_OBLIGATIONS O1
PRESERVE_OBLIGATIONS O2
NON_GOAL_OBLIGATIONS O3
UNIT U1
OBLIGATIONS O1,O2
OP REPLACE
PATH pytboss/http.py
NODE N-8C50528DDB0A86E2
NODE_SHA 5ebb5961a4572f3b29de42aa098cb5e030675f3cd4df922fa84c9226e2327e3e
<<<
            self._connected = False
            raise NotConnectedError(f"Connection timed out: {ex}") from ex
>>>
END_UNIT
LOSS NONE
END

## Gates
- `local_model_contract_ready`: passed=`True` severity=`hard`
- `direct_generation_request_ready`: passed=`True` severity=`hard`
- `direct_mode_has_no_effect_authority`: passed=`True` severity=`hard`
- `local_inference_completed`: passed=`True` severity=`hard`
- `live_route_integrity_release_authorized`: passed=`True` severity=`hard`
- `raw_text_training_disabled`: passed=`True` severity=`hard`
- `runtime_external_inference_disabled`: passed=`True` severity=`hard`
- `public_benchmark_training_disabled`: passed=`True` severity=`hard`
- `fallback_returns_disabled`: passed=`True` severity=`hard`
