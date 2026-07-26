from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import macos_mlx_environment_diagnosis as diagnosis  # noqa: E402


def test_no_metal_context_does_not_recommend_reinstalling_healthy_mlx() -> None:
    active = {
        "python": "qualified-python",
        "mlx_core_usable": False,
        "native_abort": False,
        "metal_unavailable": True,
    }

    route = diagnosis.route_decision([], [], [active], active)

    assert route["action"] == "disable_mlx_acceleration_route"
    assert route["execution_context_fault"] == (
        "metal_device_unavailable_in_current_context"
    )
    assert "governed host-Metal runner" in route["smallest_safe_fix"]
    assert "reinstall" in route["smallest_safe_fix"]
    assert "Do not reinstall" in route["smallest_safe_fix"]


def test_native_abort_without_context_fault_retains_clean_runtime_repair() -> None:
    active = {
        "python": "broken-python",
        "mlx_core_usable": False,
        "native_abort": True,
        "metal_unavailable": False,
    }

    route = diagnosis.route_decision([], [active], [], active)

    assert route["action"] == "disable_mlx_acceleration_route"
    assert "clean Apple-Silicon MLX runtime" in route["smallest_safe_fix"]


def test_usable_runtime_still_wins_route_selection() -> None:
    active = {
        "python": "usable-python",
        "mlx_core_usable": True,
        "native_abort": False,
        "metal_unavailable": False,
    }

    route = diagnosis.route_decision([active], [], [], active)

    assert route["action"] == "route_mlx_to_usable_python"
    assert route["recommended_python"] == "usable-python"


def test_canonical_qualified_mlx_runtime_is_a_probe_candidate() -> None:
    candidates = diagnosis.candidate_pythons()

    assert (
        ROOT / "runtime" / "venvs" / "mlx-0.32.0-py312" / "bin" / "python"
    ) in candidates
