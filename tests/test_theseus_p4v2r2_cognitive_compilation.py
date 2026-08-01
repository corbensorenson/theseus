from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4v2r2_cognitive_compilation as p4v  # noqa: E402


INSTRUMENT = ROOT / "configs/theseus_p4v2r2_cognitive_compilation_instrument.json"


def test_consumed_predecessor_instrument_stays_bound_to_its_original_runner() -> None:
    report = p4v.audit_instrument(INSTRUMENT)

    assert report["trigger_state"] == "RED"
    assert report["faults"] == ["candidate_runner_digest_mismatch"]
    assert report["v2r2_mechanics"]["state"] == (
        "PROSPECTIVE_LIST_NORMALIZATION_MECHANICS_GREEN"
    )
    assert report["completion_model_contract"]["ready"] is True


def test_instrument_holds_tmax_and_generation_completion_constant() -> None:
    value = json.loads(INSTRUMENT.read_text(encoding="utf-8"))
    base = json.loads(
        (ROOT / value["base_local_instrument"]).read_text(encoding="utf-8")
    )

    assert base["frozen_model"]["repo_id"] == "mlx-community/Tmax-9B-MLX-8bit"
    assert base["frozen_model"]["project_selected_quality_token_cap"] is None
    assert value["generation_budget"]["project_selected_quality_token_cap"] is None
    assert value["matched_arm_contract"]["same_frozen_weights"] is True
    assert value["matched_arm_contract"]["same_completion_policy"] is True
    assert value["fresh_task_pool_contract"]["candidate_generation_opened"] is False
    assert value["boundaries"]["user_gate"] == "none"


def test_semantic_prompt_and_completion_use_v2r2() -> None:
    prompt = p4v.render_arm_prompt(
        p4.SEMANTIC,
        {"natural_request": "change behavior"},
        "[INFORMATION_MATCHED_OBLIGATIONS]\nO1 REQUIRE: change",
        {},
    )
    candidate = (
        "THESEUS_SEMANTIC_IR_V2\nSOURCE "
        + "a" * 64
        + "\nALL_OBLIGATIONS ['O1']\nUNIT U1\nOBLIGATIONS O1\nOP REPLACE\n"
        "PATH sample.py\nNODE N-ONE\nNODE_SHA "
        + "b" * 64
        + "\n<<<\nx = 2\n>>>\nEND_UNIT\nLOSS NONE\nEND"
    )

    assert prompt.startswith(p4v.SEMANTIC_PROMPT_MARKER)
    assert "theseus_semantic_ir_v2r2" in p4v.ir_v2r2.__name__
    assert p4v.candidate_envelope_complete(candidate)


def test_persistent_session_uses_v2_decoder_and_contract(monkeypatch) -> None:
    captured = {}

    class Session:
        faults = []
        contract = {}
        identity = {}
        boundary = False

        def generate_report(self, **_kwargs):
            return {
                "policy": "v1",
                "trigger_state": "GREEN",
                "faults": [],
                "response": {"answer": "ok", "evidence": {"backend_policy": "v1"}},
                "metrics": {"physical_context_boundary_hit": self.boundary},
            }

    def persistent(**kwargs):
        captured.update(kwargs)
        return Session()

    sentinel = object()

    def model(card, snapshot, maximum, *, completion_predicate):
        captured["v2_model"] = (card, snapshot, maximum, completion_predicate)
        return sentinel

    contract = {"ready": True, "faults": [], "identity": {"identity_sha256": "x"}}
    monkeypatch.setattr(p4v.backend_v1, "PersistentLocalInferenceSession", persistent)
    monkeypatch.setattr(p4v.backend_v2, "LocalMlxChatModel", model)
    monkeypatch.setattr(
        p4v.backend_v2.route_integrity, "load_model_contract", lambda *args, **kwargs: contract
    )
    predicate = lambda text: text.endswith("END")
    session = p4v.persistent_v2_session(
        worker_config_path=Path("worker.json"),
        runtime_preflight_path=Path("preflight.json"),
        maximum_tokens=262144,
        completion_predicate=predicate,
    )
    created = captured["model_factory"]({}, Path("snapshot"), 262144)

    assert created is sentinel
    assert captured["v2_model"][3] is predicate
    assert session.contract is contract
    assert session.identity == contract["identity"]
    report = session.generate_report(prompt="test")
    assert report["policy"] == p4v.backend_v2.BACKEND_POLICY
    assert report["response"]["evidence"]["backend_policy"] == (
        p4v.backend_v2.BACKEND_POLICY
    )
    session.boundary = True
    invalid = session.generate_report(prompt="test")
    assert invalid["trigger_state"] == "RED"
    assert invalid["response"]["answer"] == ""
    assert "instrument_inadequate_generation_boundary_hit" in invalid["faults"]


def test_route_guard_contract_is_source_present_and_fail_fast() -> None:
    source = (ROOT / "scripts" / "theseus_p4v2r2_cognitive_compilation.py").read_text(
        encoding="utf-8"
    )

    assert "route_guarded_runtime_call" in source
    assert "p4v2r2_route_integrity_release_failed" in source
    assert 'route_receipt.get("release_allowed") is not True' in source
