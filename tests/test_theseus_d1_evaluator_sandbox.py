from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_d1_evaluator_sandbox as sandbox  # noqa: E402


CONFIG_PATH = ROOT / "configs/theseus_d1_evaluator_sandbox.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_contract_is_prospective_and_grants_no_execution_by_itself() -> None:
    value = config()
    assert sandbox.validate_config(value) == []
    report = sandbox.preflight(value, config_path=CONFIG_PATH)
    assert report["trigger_state"] == "PAUSED"
    assert report["untrusted_execution_authorized"] is False
    assert report["parent_target_or_evaluator_executions"] == 0
    assert report["candidate_or_control_calls"] == 0


def test_profile_denies_confidentiality_effect_and_exec_escape_surfaces(tmp_path: Path) -> None:
    profile = sandbox.sandbox_profile(config(), Path("/private/tmp/theseus-d1-test"))
    assert "(deny network*)" in profile
    assert "(deny mach-lookup)" in profile
    assert '(subpath "/Users")' in profile
    assert '(subpath "/private/var/folders")' in profile
    assert "deny file-write*" in profile
    assert "deny process-exec" in profile
    assert "/Library/Frameworks/Python.framework/Versions/3.12" in profile


def test_run_rejects_non_pinned_python_or_home_workdir(tmp_path: Path) -> None:
    value = config()
    try:
        sandbox.run_sandboxed(["/bin/sh"], workdir=Path("/private/tmp/x"), config=value)
    except ValueError as exc:
        assert str(exc) == "sandbox_command_not_pinned_python"
    else:
        raise AssertionError("non-Python command accepted")
    try:
        sandbox.run_sandboxed([value["python"]], workdir=ROOT, config=value)
    except ValueError as exc:
        assert str(exc) == "sandbox_workdir_outside_required_root"
    else:
        raise AssertionError("home workdir accepted")


def test_canary_parser_requires_exact_all_true_keyset() -> None:
    required = {
        "host_read_denied",
        "symlink_host_read_denied",
        "outside_write_denied",
        "network_denied",
        "shell_exec_denied",
        "child_python_inherits_read_denial",
        "inside_write_allowed",
        "environment_minimized",
        "cpu_limit_present",
        "file_size_limit_present",
        "open_file_limit_present",
        "process_limit_present",
    }
    good = {
        "stdout_tail": sandbox.RESULT_MARKER + json.dumps({key: True for key in required})
    }
    value, faults = sandbox.parse_canary(good)
    assert faults == []
    assert all(value.values())
    bad_payload = {key: True for key in required}
    bad_payload["network_denied"] = False
    _, faults = sandbox.parse_canary(
        {"stdout_tail": sandbox.RESULT_MARKER + json.dumps(bad_payload)}
    )
    assert "sandbox_canary_failed:network_denied" in faults
    extra_payload = {key: True for key in required}
    extra_payload["producer_says_safe"] = True
    _, faults = sandbox.parse_canary(
        {"stdout_tail": sandbox.RESULT_MARKER + json.dumps(extra_payload)}
    )
    assert "sandbox_canary_keyset_invalid" in faults
