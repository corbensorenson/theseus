import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vcm_sandbox", ROOT / "scripts/theseus_vcm_multilang_sandbox.py")
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MOD)


def test_preflight_binds_exact_tools_and_denies_repository_execution():
    path = ROOT / "configs/theseus_vcm_multilang_sandbox.json"
    report = MOD.preflight(json.loads(path.read_text()), path)
    assert report["trigger_state"] == "PAUSED"
    assert report["repository_execution_authorized"] is False
    assert report["candidate_or_control_calls"] == 0


def test_parser_requires_exact_true_canary_keyset():
    good = {key: True for key in MOD.REQUIRED}
    parsed, faults = MOD.parse({"stdout": MOD.MARKER + json.dumps(good) + "\n"})
    assert parsed == good
    assert faults == []
    bad = dict(good)
    bad["network_denied"] = False
    _parsed, faults = MOD.parse({"stdout": MOD.MARKER + json.dumps(bad) + "\n"})
    assert "canary_failed:network_denied" in faults


def test_profile_denies_network_and_limits_process_exec():
    config = json.loads((ROOT / "configs/theseus_vcm_multilang_sandbox.json").read_text())
    profile = MOD.sandbox_profile(config["executables"]["node"]["path"], Path("/private/tmp/vcm-test"), config, "node")
    assert "(deny network*)" in profile
    assert "deny process-exec" in profile
    assert "/Users" in profile
