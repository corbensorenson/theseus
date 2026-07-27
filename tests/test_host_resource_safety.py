from __future__ import annotations

import inspect
import signal
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import host_resource_safety as safety  # noqa: E402


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 1000.
Pages active: 2000.
Pages inactive: 3000.
Pages speculative: 400.
Pages wired down: 500.
Pages purgeable: 600.
Swapouts: 700.
"""


def test_vm_stat_parser_tracks_reclaimable_memory_and_swap() -> None:
    snapshot = safety.parse_vm_stat(VM_STAT)
    assert snapshot.page_size_bytes == 16384
    assert snapshot.reclaimable_available_mib == pytest.approx(78.125)
    assert snapshot.swapouts_mib == pytest.approx(10.9375)


def test_policy_rejects_a_process_limit_above_half_of_host_memory() -> None:
    policy = safety.HostSafetyPolicy(
        max_process_memory_mib=9000,
        minimum_available_before_launch_mib=8000,
        minimum_available_during_run_mib=4000,
        maximum_swapout_growth_mib=64,
        maximum_wall_seconds=60,
    )
    with pytest.raises(safety.HostResourceSafetyFault, match="half_physical"):
        policy.validate(physical_memory_mib=16384)


def test_policy_overrides_size_guard_to_workload_without_hidden_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(safety, "physical_memory_mib", lambda: 16384.0)
    default = safety.default_policy(maximum_wall_seconds=60)
    policy = safety.policy_with_overrides(
        default,
        max_process_memory_mib=1024,
        minimum_available_before_launch_mib=2048,
        minimum_available_during_run_mib=1024,
        maximum_swapout_growth_mib=16,
    )

    assert policy.max_process_memory_mib == 1024
    assert policy.minimum_available_before_launch_mib == 2048
    assert policy.minimum_available_during_run_mib == 1024
    assert policy.maximum_swapout_growth_mib == 16
    assert policy.maximum_wall_seconds == default.maximum_wall_seconds
    assert policy.poll_interval_seconds == default.poll_interval_seconds


def test_explicit_zero_reserve_floor_is_observation_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(safety, "physical_memory_mib", lambda: 16384.0)
    policy = safety.policy_with_overrides(
        safety.default_policy(maximum_wall_seconds=60),
        minimum_available_before_launch_mib=0,
        minimum_available_during_run_mib=0,
        maximum_swapout_growth_mib=0,
    )

    assert policy.minimum_available_before_launch_mib == 0
    assert policy.minimum_available_during_run_mib == 0
    assert policy.maximum_swapout_growth_mib == 0


def test_mapping_preserves_explicit_zero_and_predicted_exhaustion_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(safety, "physical_memory_mib", lambda: 16384.0)
    policy = safety.policy_from_mapping(
        {
            "minimum_available_before_launch_mib": 0,
            "minimum_available_during_run_mib": 0,
            "memory_guard_mode": "predicted_exhaustion",
            "maximum_swapout_growth_mib": 0,
        },
        maximum_wall_seconds=0,
    )
    assert policy.minimum_available_before_launch_mib == 0
    assert policy.minimum_available_during_run_mib == 0
    assert policy.memory_guard_mode == "predicted_exhaustion"
    assert policy.maximum_wall_seconds == 0


def test_predicted_exhaustion_mode_rejects_a_hidden_fixed_floor() -> None:
    policy = safety.HostSafetyPolicy(
        max_process_memory_mib=512,
        minimum_available_before_launch_mib=1,
        minimum_available_during_run_mib=0,
        maximum_swapout_growth_mib=0,
        maximum_wall_seconds=0,
        memory_guard_mode="predicted_exhaustion",
    )
    with pytest.raises(
        safety.HostResourceSafetyFault,
        match="cannot_hide_a_fixed_reserve",
    ):
        policy.validate(physical_memory_mib=16384)


def test_negative_reserve_threshold_remains_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(safety, "physical_memory_mib", lambda: 16384.0)
    with pytest.raises(safety.HostResourceSafetyFault, match="negative_threshold"):
        safety.policy_with_overrides(
            safety.default_policy(maximum_wall_seconds=60),
            minimum_available_before_launch_mib=-1,
        )


def test_policy_overrides_still_reject_invalid_reserve_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(safety, "physical_memory_mib", lambda: 16384.0)
    with pytest.raises(safety.HostResourceSafetyFault, match="launch_reserve"):
        safety.policy_with_overrides(
            safety.default_policy(maximum_wall_seconds=60),
            minimum_available_before_launch_mib=1024,
            minimum_available_during_run_mib=2048,
        )


def test_process_group_rss_sums_parent_and_nested_children() -> None:
    table = """101 101 1024
102 101 2048
103 101 3072
201 201 9999
"""
    assert safety.parse_process_group_rss_mib(
        table, process_group_id=101
    ) == pytest.approx(6.0)
    with pytest.raises(safety.HostResourceSafetyFault, match="group_unavailable"):
        safety.parse_process_group_rss_mib(table, process_group_id=999)


def test_interrupt_cleanup_terminates_child_group_and_restores_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: dict[int, object] = {}
    terminated: list[tuple[object, float]] = []

    class Process:
        def poll(self) -> None:
            return None

    process = Process()
    previous = signal.default_int_handler
    monkeypatch.setattr(safety.signal, "getsignal", lambda _sig: previous)
    monkeypatch.setattr(
        safety.signal,
        "signal",
        lambda sig, handler: installed.__setitem__(sig, handler),
    )
    monkeypatch.setattr(
        safety,
        "_terminate_process_group",
        lambda observed, grace: terminated.append((observed, grace)),
    )

    restore = safety._install_interrupt_cleanup(process, 2.0)
    handler = installed[signal.SIGINT]
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGINT, None)

    assert terminated == [(process, 2.0)]
    assert installed[signal.SIGINT] is previous
    restore()
    assert terminated == [(process, 2.0)]


def test_guard_terminates_child_before_memory_or_swap_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots = iter(
        [
            safety.HostMemorySnapshot(16384, 12000, 10, 16384),
            safety.HostMemorySnapshot(16384, 11000, 10, 16384),
            safety.HostMemorySnapshot(16384, 10000, 10, 16384),
        ]
    )
    policy = safety.HostSafetyPolicy(
        max_process_memory_mib=64,
        minimum_available_before_launch_mib=8000,
        minimum_available_during_run_mib=4000,
        maximum_swapout_growth_mib=64,
        maximum_wall_seconds=10,
        poll_interval_seconds=0.01,
        terminate_grace_seconds=1,
    )
    result = safety.run_guarded(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=ROOT,
        policy=policy,
        snapshot_fn=lambda: next(snapshots),
        rss_fn=lambda _pid: 65.0,
    )
    assert result.receipt["passed"] is False
    assert result.receipt["child_started"] is True
    assert result.receipt["terminated_by_guard"] is True
    assert result.receipt["fault"] == "process_memory_limit_exceeded"


def test_guard_records_reclaimable_drop_as_unified_memory_diagnostic() -> None:
    calls = 0

    def snapshot() -> safety.HostMemorySnapshot:
        nonlocal calls
        calls += 1
        available = 9000 if calls == 1 else 8300
        return safety.HostMemorySnapshot(16384, available, 10, 16384)

    policy = safety.HostSafetyPolicy(
        max_process_memory_mib=512,
        minimum_available_before_launch_mib=6000,
        minimum_available_during_run_mib=3000,
        maximum_swapout_growth_mib=16,
        maximum_wall_seconds=10,
        poll_interval_seconds=0.01,
        terminate_grace_seconds=1,
    )
    result = safety.run_guarded(
        [sys.executable, "-c", "import time; time.sleep(.03)"],
        cwd=ROOT,
        policy=policy,
        snapshot_fn=snapshot,
        rss_fn=lambda _pid: 64.0,
    )
    assert result.receipt["passed"] is True
    assert result.receipt["maximum_process_rss_mib"] == 64.0
    assert result.receipt["maximum_inferred_unified_memory_mib"] == 700.0


def test_guard_retains_terminal_swap_observation_after_prefix_is_full() -> None:
    calls = 0

    def snapshot() -> safety.HostMemorySnapshot:
        nonlocal calls
        calls += 1
        swapouts = 10 if calls < 70 else 27
        return safety.HostMemorySnapshot(16384, 9000, swapouts, 16384)

    policy = safety.HostSafetyPolicy(
        max_process_memory_mib=512,
        minimum_available_before_launch_mib=6000,
        minimum_available_during_run_mib=2000,
        maximum_swapout_growth_mib=16,
        maximum_wall_seconds=10,
        poll_interval_seconds=0.001,
        terminate_grace_seconds=1,
    )
    result = safety.run_guarded(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=ROOT,
        policy=policy,
        snapshot_fn=snapshot,
        rss_fn=lambda _pid: 64.0,
    )
    receipt = result.receipt
    assert receipt["fault"] == "swap_growth_limit_exceeded"
    assert len(receipt["observation_prefix"]) == 64
    assert receipt["observation_prefix"][-1]["swapout_growth_mib"] == 0.0
    assert receipt["observation_suffix"][-1]["swapout_growth_mib"] == 17
    assert receipt["fault_observation"] == receipt["observation_suffix"][-1]


def test_guard_can_report_system_wide_swap_growth_without_attributing_it_to_child() -> None:
    calls = 0

    def snapshot() -> safety.HostMemorySnapshot:
        nonlocal calls
        calls += 1
        swapouts = 10 if calls < 3 else 40
        return safety.HostMemorySnapshot(16384, 9000, swapouts, 16384)

    policy = safety.HostSafetyPolicy(
        max_process_memory_mib=512,
        minimum_available_before_launch_mib=6000,
        minimum_available_during_run_mib=2000,
        maximum_swapout_growth_mib=16,
        maximum_wall_seconds=10,
        poll_interval_seconds=0.001,
        terminate_grace_seconds=1,
        swapout_growth_action="report_only",
    )
    result = safety.run_guarded(
        [sys.executable, "-c", "import time; time.sleep(0.03)"],
        cwd=ROOT,
        policy=policy,
        snapshot_fn=snapshot,
        rss_fn=lambda _pid: 64.0,
    )
    receipt = result.receipt
    assert receipt["passed"] is True
    assert receipt["fault"] == ""
    assert receipt["swapout_growth_action"] == "report_only"
    assert receipt["swapout_growth_threshold_exceeded"] is True
    assert receipt["swapout_growth_breach_observations"] >= 1
    assert (
        receipt["first_swapout_growth_breach_observation"][
            "swapout_growth_mib"
        ]
        == 30
    )


def test_cli_exposes_report_only_swap_action() -> None:
    source = inspect.getsource(safety.main)

    assert '"--swapout-growth-action"' in source
    assert "choices=sorted(SWAPOUT_GROWTH_ACTIONS)" in source
    assert "swapout_growth_action=args.swapout_growth_action" in source


def test_guard_tolerates_one_transient_reserve_sample() -> None:
    calls = 0

    def snapshot() -> safety.HostMemorySnapshot:
        nonlocal calls
        calls += 1
        available = 9000 if calls == 1 else 1999 if calls == 2 else 2500
        return safety.HostMemorySnapshot(16384, available, 10, 16384)

    policy = safety.HostSafetyPolicy(
        max_process_memory_mib=512,
        minimum_available_before_launch_mib=6000,
        minimum_available_during_run_mib=2000,
        maximum_swapout_growth_mib=16,
        maximum_wall_seconds=10,
        poll_interval_seconds=0.01,
        terminate_grace_seconds=1,
    )
    result = safety.run_guarded(
        [sys.executable, "-c", "import time; time.sleep(.04)"],
        cwd=ROOT,
        policy=policy,
        snapshot_fn=snapshot,
        rss_fn=lambda _pid: 64.0,
    )
    assert result.receipt["passed"] is True
    assert result.receipt["maximum_consecutive_reserve_breaches"] == 1


def test_guard_terminates_on_three_consecutive_reserve_samples() -> None:
    calls = 0

    def snapshot() -> safety.HostMemorySnapshot:
        nonlocal calls
        calls += 1
        available = 9000 if calls == 1 else 1900
        return safety.HostMemorySnapshot(16384, available, 10, 16384)

    policy = safety.HostSafetyPolicy(
        max_process_memory_mib=512,
        minimum_available_before_launch_mib=6000,
        minimum_available_during_run_mib=2000,
        maximum_swapout_growth_mib=16,
        maximum_wall_seconds=10,
        poll_interval_seconds=0.01,
        terminate_grace_seconds=1,
    )
    result = safety.run_guarded(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=ROOT,
        policy=policy,
        snapshot_fn=snapshot,
        rss_fn=lambda _pid: 64.0,
    )
    assert result.receipt["passed"] is False
    assert result.receipt["fault"] == "host_memory_reserve_breached"
    assert result.receipt["maximum_consecutive_reserve_breaches"] == 3


def test_predicted_exhaustion_guard_uses_measured_decline_not_a_floor() -> None:
    calls = 0

    def snapshot() -> safety.HostMemorySnapshot:
        nonlocal calls
        calls += 1
        available = [9000, 2000, 1000, 500, 250][min(calls - 1, 4)]
        return safety.HostMemorySnapshot(16384, available, 10, 16384)

    policy = safety.HostSafetyPolicy(
        max_process_memory_mib=512,
        minimum_available_before_launch_mib=0,
        minimum_available_during_run_mib=0,
        maximum_swapout_growth_mib=0,
        maximum_wall_seconds=0,
        poll_interval_seconds=0.01,
        terminate_grace_seconds=1,
        swapout_growth_action="report_only",
        memory_guard_mode="predicted_exhaustion",
    )
    result = safety.run_guarded(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=ROOT,
        policy=policy,
        snapshot_fn=snapshot,
        rss_fn=lambda _pid: 64.0,
    )
    assert result.receipt["passed"] is False
    assert result.receipt["fault"] == "host_memory_exhaustion_predicted"
    assert result.receipt["minimum_reclaimable_available_mib"] == 500
    assert result.receipt["memory_guard_mode"] == "predicted_exhaustion"
    assert result.receipt["wall_limit_disabled"] is True


def test_predicted_exhaustion_guard_does_not_kill_a_stable_low_plateau() -> None:
    calls = 0

    def snapshot() -> safety.HostMemorySnapshot:
        nonlocal calls
        calls += 1
        available = 9000 if calls == 1 else 500
        return safety.HostMemorySnapshot(16384, available, 10, 16384)

    policy = safety.HostSafetyPolicy(
        max_process_memory_mib=512,
        minimum_available_before_launch_mib=0,
        minimum_available_during_run_mib=0,
        maximum_swapout_growth_mib=0,
        maximum_wall_seconds=0,
        poll_interval_seconds=0.01,
        terminate_grace_seconds=1,
        swapout_growth_action="report_only",
        memory_guard_mode="predicted_exhaustion",
    )
    result = safety.run_guarded(
        [sys.executable, "-c", "import time; time.sleep(.06)"],
        cwd=ROOT,
        policy=policy,
        snapshot_fn=snapshot,
        rss_fn=lambda _pid: 64.0,
    )
    assert result.receipt["passed"] is True
    assert (
        result.receipt[
            "maximum_consecutive_predicted_exhaustion_observations"
        ]
        == 1
    )


def test_qualified_working_set_suppresses_a_known_finite_allocation_ramp() -> None:
    calls = 0

    def snapshot() -> safety.HostMemorySnapshot:
        nonlocal calls
        calls += 1
        values = [7000, 4820, 4386, 3141, 3100]
        available = values[min(calls - 1, len(values) - 1)]
        return safety.HostMemorySnapshot(16384, available, 10, 16384)

    policy = safety.HostSafetyPolicy(
        max_process_memory_mib=512,
        minimum_available_before_launch_mib=0,
        minimum_available_during_run_mib=0,
        maximum_swapout_growth_mib=0,
        maximum_wall_seconds=0,
        poll_interval_seconds=0.01,
        terminate_grace_seconds=1,
        swapout_growth_action="report_only",
        memory_guard_mode="predicted_exhaustion",
        qualified_peak_inferred_unified_memory_mib=4800,
    )
    result = safety.run_guarded(
        [sys.executable, "-c", "import time; time.sleep(.07)"],
        cwd=ROOT,
        policy=policy,
        snapshot_fn=snapshot,
        rss_fn=lambda _pid: 64.0,
    )
    assert result.receipt["passed"] is True
    assert any(
        row["within_qualified_working_set"]
        for row in result.receipt["observation_prefix"]
    )


def test_guard_tolerates_two_transient_telemetry_failures() -> None:
    calls = 0

    def snapshot() -> safety.HostMemorySnapshot:
        nonlocal calls
        calls += 1
        if calls in {2, 3}:
            raise PermissionError("transient vm_stat race")
        return safety.HostMemorySnapshot(16384, 9000, 10, 16384)

    policy = safety.HostSafetyPolicy(
        max_process_memory_mib=512,
        minimum_available_before_launch_mib=6000,
        minimum_available_during_run_mib=2000,
        maximum_swapout_growth_mib=16,
        maximum_wall_seconds=10,
        poll_interval_seconds=0.01,
        terminate_grace_seconds=1,
    )
    result = safety.run_guarded(
        [sys.executable, "-c", "import time; time.sleep(.06)"],
        cwd=ROOT,
        policy=policy,
        snapshot_fn=snapshot,
        rss_fn=lambda _pid: 64.0,
    )
    assert result.receipt["passed"] is True
    assert result.receipt["maximum_consecutive_telemetry_failures"] == 2
    assert result.receipt["telemetry_failure_observations_allowed"] == 2


def test_guard_fails_closed_on_three_consecutive_telemetry_failures() -> None:
    calls = 0

    def snapshot() -> safety.HostMemorySnapshot:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise PermissionError("persistent vm_stat failure")
        return safety.HostMemorySnapshot(16384, 9000, 10, 16384)

    policy = safety.HostSafetyPolicy(
        max_process_memory_mib=512,
        minimum_available_before_launch_mib=6000,
        minimum_available_during_run_mib=2000,
        maximum_swapout_growth_mib=16,
        maximum_wall_seconds=10,
        poll_interval_seconds=0.01,
        terminate_grace_seconds=1,
    )
    result = safety.run_guarded(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=ROOT,
        policy=policy,
        snapshot_fn=snapshot,
        rss_fn=lambda _pid: 64.0,
    )
    assert result.receipt["passed"] is False
    assert result.receipt["fault"] == (
        "safety_telemetry_unavailable:host_memory_snapshot:PermissionError"
    )
    assert result.receipt["maximum_consecutive_telemetry_failures"] == 3


def test_guard_keeps_host_pressure_active_when_process_rss_is_permission_denied() -> None:
    policy = safety.HostSafetyPolicy(
        max_process_memory_mib=512,
        minimum_available_before_launch_mib=6000,
        minimum_available_during_run_mib=2000,
        maximum_swapout_growth_mib=16,
        maximum_wall_seconds=10,
        poll_interval_seconds=0.01,
        terminate_grace_seconds=1,
    )
    result = safety.run_guarded(
        [sys.executable, "-c", "import time; time.sleep(.06)"],
        cwd=ROOT,
        policy=policy,
        snapshot_fn=lambda: safety.HostMemorySnapshot(16384, 9000, 10, 16384),
        rss_fn=lambda _pid: (_ for _ in ()).throw(PermissionError("ps denied")),
    )
    assert result.receipt["passed"] is True
    assert result.receipt["process_rss_telemetry_state"] == (
        "UNAVAILABLE_PERMISSION_HOST_PRESSURE_ACTIVE"
    )
    assert result.receipt["maximum_consecutive_telemetry_failures"] == 3
    assert result.receipt["minimum_reclaimable_available_mib"] == 9000
