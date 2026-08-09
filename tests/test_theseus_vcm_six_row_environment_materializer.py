from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_six_row_environment_materializer as owner  # noqa:E402
import theseus_vcm_six_row_environment_materializer_audit as audit_owner  # noqa:E402
CONFIG=ROOT/"configs/theseus_vcm_six_row_environment_materializer.json"
def test_materializer_preflight_binds_one_six_row_family()->None:
 cfg,bound,faults=owner.preflight(CONFIG);assert faults in ([],["predecessor_store_binding_invalid"]);assert list(bound["rows"])==[12,13,16,25,35,56];assert {r["manager"] for r in cfg["rows"]}=={"uv","cargo"}
def test_materializer_has_no_runner_evaluator_or_model_authority()->None:
 cfg=owner.p2a.read_json(CONFIG);assert cfg["authority"]["source_build_authorized"] is False;assert cfg["authority"]["repository_runner_execution_authorized"] is False;assert cfg["authority"]["parent_target_evaluator_execution_authorized"] is False;assert cfg["authority"]["local_model_calls_authorized"] is False;assert cfg["authority"]["external_reference_calls_authorized"] is False
def test_materializer_post_store_is_role_separately_green()->None:
 report=audit_owner.audit(CONFIG);assert report["trigger_state"]=="GREEN";assert report["qualified_task_count"]==6;assert report["inconclusive_task_count"]==0
