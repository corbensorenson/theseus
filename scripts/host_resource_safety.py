#!/usr/bin/env python3
"""Fail-closed host safety envelope for resource-intensive subprocesses.

This module exists because in-process, step-boundary telemetry cannot stop a
single operation whose unified-memory footprint grows beyond the host's safe
working set. Heavy accelerator and replay work must run as a child of this guard.
"""

from __future__ import annotations

import json
import argparse
import os
import re
import signal
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


POLICY = "project_theseus_host_resource_safety_v1"
MIB = 1024 * 1024
MINIMUM_CONSECUTIVE_RESERVE_BREACHES = 3
MAXIMUM_CONSECUTIVE_TELEMETRY_FAILURES = 3
SWAPOUT_GROWTH_ACTIONS = frozenset({"hard_stop", "report_only"})
VM_STAT_PATTERN = re.compile(r'^\"?([^\"]+?)\"?:\s+([0-9]+)\.?$')


class HostResourceSafetyFault(RuntimeError):
    pass


def accelerator_child_authorized() -> bool:
    """Return whether this process was launched inside the external guard."""

    return os.environ.get("THESEUS_GUARDED_ACCELERATOR_CHILD") == "1"


@dataclass(frozen=True)
class HostSafetyPolicy:
    max_process_memory_mib: float
    minimum_available_before_launch_mib: float
    minimum_available_during_run_mib: float
    maximum_swapout_growth_mib: float
    maximum_wall_seconds: float
    poll_interval_seconds: float = 0.25
    terminate_grace_seconds: float = 2.0
    swapout_growth_action: str = "hard_stop"

    def validate(self, *, physical_memory_mib: float) -> None:
        strictly_positive_values = (
            self.max_process_memory_mib,
            self.maximum_wall_seconds,
            self.poll_interval_seconds,
            self.terminate_grace_seconds,
        )
        nonnegative_thresholds = (
            self.minimum_available_before_launch_mib,
            self.minimum_available_during_run_mib,
            self.maximum_swapout_growth_mib,
        )
        if any(float(value) <= 0 for value in strictly_positive_values):
            raise HostResourceSafetyFault("host_safety_policy_nonpositive")
        if any(float(value) < 0 for value in nonnegative_thresholds):
            raise HostResourceSafetyFault("host_safety_policy_negative_threshold")
        if self.swapout_growth_action not in SWAPOUT_GROWTH_ACTIONS:
            raise HostResourceSafetyFault("swapout_growth_action_invalid")
        if self.max_process_memory_mib > physical_memory_mib * 0.5:
            raise HostResourceSafetyFault("process_limit_exceeds_half_physical_memory")
        if self.minimum_available_before_launch_mib < self.minimum_available_during_run_mib:
            raise HostResourceSafetyFault("launch_reserve_below_runtime_reserve")
        if self.poll_interval_seconds > 1.0:
            raise HostResourceSafetyFault("host_safety_poll_interval_too_slow")


@dataclass(frozen=True)
class HostMemorySnapshot:
    physical_memory_mib: float
    reclaimable_available_mib: float
    swapouts_mib: float
    page_size_bytes: int


@dataclass(frozen=True)
class GuardedProcessResult:
    returncode: int
    stdout: str
    stderr: str
    receipt: dict[str, Any]


def physical_memory_mib() -> float:
    return float(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / MIB


def parse_vm_stat(text: str) -> HostMemorySnapshot:
    header = re.search(r"page size of ([0-9]+) bytes", text)
    if not header:
        raise HostResourceSafetyFault("vm_stat_page_size_missing")
    page_size = int(header.group(1))
    values: dict[str, int] = {}
    for line in text.splitlines()[1:]:
        match = VM_STAT_PATTERN.match(line.strip())
        if match:
            values[match.group(1)] = int(match.group(2))
    required = {
        "Pages free",
        "Pages inactive",
        "Pages speculative",
        "Pages purgeable",
        "Swapouts",
    }
    missing = sorted(required - set(values))
    if missing:
        raise HostResourceSafetyFault("vm_stat_fields_missing:" + ",".join(missing))
    reclaimable_pages = sum(
        values[key]
        for key in (
            "Pages free",
            "Pages inactive",
            "Pages speculative",
            "Pages purgeable",
        )
    )
    return HostMemorySnapshot(
        physical_memory_mib=physical_memory_mib(),
        reclaimable_available_mib=reclaimable_pages * page_size / MIB,
        swapouts_mib=values["Swapouts"] * page_size / MIB,
        page_size_bytes=page_size,
    )


def host_memory_snapshot() -> HostMemorySnapshot:
    process = subprocess.run(
        ["/usr/bin/vm_stat"],
        text=True,
        capture_output=True,
        timeout=2.0,
        check=False,
    )
    if process.returncode != 0:
        raise HostResourceSafetyFault(
            "vm_stat_unavailable:" + process.stderr.strip()[-300:]
        )
    return parse_vm_stat(process.stdout)


def default_policy(*, maximum_wall_seconds: float) -> HostSafetyPolicy:
    physical = physical_memory_mib()
    return HostSafetyPolicy(
        max_process_memory_mib=min(6144.0, physical * 0.375),
        minimum_available_before_launch_mib=min(6144.0, physical * 0.375),
        minimum_available_during_run_mib=min(4096.0, physical * 0.25),
        maximum_swapout_growth_mib=64.0,
        maximum_wall_seconds=maximum_wall_seconds,
        poll_interval_seconds=0.25,
        terminate_grace_seconds=2.0,
    )


def policy_with_overrides(
    policy: HostSafetyPolicy,
    *,
    max_process_memory_mib: float | None = None,
    minimum_available_before_launch_mib: float | None = None,
    minimum_available_during_run_mib: float | None = None,
    maximum_swapout_growth_mib: float | None = None,
    swapout_growth_action: str | None = None,
) -> HostSafetyPolicy:
    """Apply explicit workload-sized limits without weakening validation."""

    overridden = HostSafetyPolicy(
        max_process_memory_mib=(
            policy.max_process_memory_mib
            if max_process_memory_mib is None
            else float(max_process_memory_mib)
        ),
        minimum_available_before_launch_mib=(
            policy.minimum_available_before_launch_mib
            if minimum_available_before_launch_mib is None
            else float(minimum_available_before_launch_mib)
        ),
        minimum_available_during_run_mib=(
            policy.minimum_available_during_run_mib
            if minimum_available_during_run_mib is None
            else float(minimum_available_during_run_mib)
        ),
        maximum_swapout_growth_mib=(
            policy.maximum_swapout_growth_mib
            if maximum_swapout_growth_mib is None
            else float(maximum_swapout_growth_mib)
        ),
        maximum_wall_seconds=policy.maximum_wall_seconds,
        poll_interval_seconds=policy.poll_interval_seconds,
        terminate_grace_seconds=policy.terminate_grace_seconds,
        swapout_growth_action=(
            policy.swapout_growth_action
            if swapout_growth_action is None
            else str(swapout_growth_action)
        ),
    )
    overridden.validate(physical_memory_mib=physical_memory_mib())
    return overridden


def policy_from_mapping(
    value: dict[str, Any], *, maximum_wall_seconds: float
) -> HostSafetyPolicy:
    physical = physical_memory_mib()
    fraction = float(value.get("maximum_process_physical_memory_fraction") or 0.375)
    hard_cap = float(value.get("hard_maximum_process_memory_mib") or 6144.0)
    policy = HostSafetyPolicy(
        max_process_memory_mib=min(hard_cap, physical * fraction),
        minimum_available_before_launch_mib=float(
            value.get("minimum_available_before_launch_mib") or 6144.0
        ),
        minimum_available_during_run_mib=float(
            value.get("minimum_available_during_run_mib") or 4096.0
        ),
        maximum_swapout_growth_mib=float(
            value.get("maximum_swapout_growth_mib") or 64.0
        ),
        maximum_wall_seconds=float(maximum_wall_seconds),
        poll_interval_seconds=float(value.get("poll_interval_seconds") or 0.25),
        terminate_grace_seconds=float(value.get("terminate_grace_seconds") or 2.0),
        swapout_growth_action=str(
            value.get("swapout_growth_action") or "hard_stop"
        ),
    )
    policy.validate(physical_memory_mib=physical)
    return policy


def parse_process_group_rss_mib(text: str, *, process_group_id: int) -> float:
    rss_kib = 0.0
    matched = 0
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            _pid, pgid, rss = (int(value) for value in fields)
        except ValueError:
            continue
        if pgid == process_group_id:
            rss_kib += float(rss)
            matched += 1
    if not matched:
        raise HostResourceSafetyFault("child_process_group_unavailable")
    return rss_kib / 1024.0


def process_rss_mib(pid: int) -> float:
    process = subprocess.run(
        ["/bin/ps", "-axo", "pid=,pgid=,rss="],
        text=True,
        capture_output=True,
        timeout=2.0,
        check=False,
    )
    if process.returncode != 0 or not process.stdout.strip():
        raise HostResourceSafetyFault("child_rss_unavailable")
    return parse_process_group_rss_mib(process.stdout, process_group_id=pid)


def preflight(policy: HostSafetyPolicy) -> HostMemorySnapshot:
    snapshot = host_memory_snapshot()
    policy.validate(physical_memory_mib=snapshot.physical_memory_mib)
    if snapshot.reclaimable_available_mib < policy.minimum_available_before_launch_mib:
        raise HostResourceSafetyFault(
            "host_memory_preflight_failed:"
            f"available_mib={snapshot.reclaimable_available_mib:.1f}:"
            f"required_mib={policy.minimum_available_before_launch_mib:.1f}"
        )
    return snapshot


def _terminate_process_group(process: subprocess.Popen[str], grace_seconds: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=grace_seconds)


def _install_interrupt_cleanup(
    process: subprocess.Popen[str], grace_seconds: float
) -> Callable[[], None]:
    """Ensure an interrupted watchdog never leaves its child session orphaned."""

    previous = signal.getsignal(signal.SIGINT)
    restored = False

    def restore() -> None:
        nonlocal restored
        if not restored:
            signal.signal(signal.SIGINT, previous)
            restored = True

    def handle_interrupt(signum: int, frame: Any) -> None:
        restore()
        if process.poll() is None:
            _terminate_process_group(process, grace_seconds)
        if callable(previous):
            previous(signum, frame)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_interrupt)
    return restore


def run_guarded(
    command: Sequence[str],
    *,
    cwd: str | Path,
    policy: HostSafetyPolicy,
    env: dict[str, str] | None = None,
    snapshot_fn: Callable[[], HostMemorySnapshot] = host_memory_snapshot,
    rss_fn: Callable[[int], float] = process_rss_mib,
) -> GuardedProcessResult:
    initial = preflight(policy) if snapshot_fn is host_memory_snapshot else snapshot_fn()
    policy.validate(physical_memory_mib=initial.physical_memory_mib)
    if initial.reclaimable_available_mib < policy.minimum_available_before_launch_mib:
        raise HostResourceSafetyFault("host_memory_preflight_failed")
    started = time.monotonic()
    observation_prefix: list[dict[str, Any]] = []
    observation_suffix: deque[dict[str, Any]] = deque(maxlen=64)
    fault_observation: dict[str, Any] | None = None
    maximum_rss = 0.0
    maximum_inferred_unified_memory = 0.0
    minimum_available = initial.reclaimable_available_mib
    maximum_swap_growth = 0.0
    swap_growth_breach_observations = 0
    first_swap_growth_breach_observation: dict[str, Any] | None = None
    reserve_breach_streak = 0
    maximum_reserve_breach_streak = 0
    telemetry_failure_streak = 0
    maximum_telemetry_failure_streak = 0
    telemetry_failure_prefix: list[dict[str, Any]] = []
    rss_permission_failure_streak = 0
    rss_telemetry_state = "AVAILABLE"
    fault = ""
    child_env = os.environ.copy()
    child_env.update(env or {})
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file, tempfile.TemporaryFile(
        mode="w+t", encoding="utf-8"
    ) as stderr_file:
        process = subprocess.Popen(
            [str(value) for value in command],
            cwd=str(cwd),
            env=child_env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            start_new_session=True,
        )
        restore_interrupt_handler = _install_interrupt_cleanup(
            process, policy.terminate_grace_seconds
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            try:
                snapshot = snapshot_fn()
            except (HostResourceSafetyFault, OSError, subprocess.SubprocessError) as exc:
                telemetry_source = "host_memory_snapshot"
                telemetry_fault_type = type(exc).__name__
                telemetry_failure_streak += 1
                maximum_telemetry_failure_streak = max(
                    maximum_telemetry_failure_streak, telemetry_failure_streak
                )
                if len(telemetry_failure_prefix) < 8:
                    telemetry_failure_prefix.append(
                        {
                            "elapsed_seconds": round(elapsed, 3),
                            "source": telemetry_source,
                            "fault_type": telemetry_fault_type,
                        }
                    )
                if elapsed > policy.maximum_wall_seconds:
                    fault = "wall_limit_exceeded"
                    break
                if (
                    telemetry_failure_streak
                    >= MAXIMUM_CONSECUTIVE_TELEMETRY_FAILURES
                ):
                    fault = (
                        f"safety_telemetry_unavailable:{telemetry_source}:"
                        f"{telemetry_fault_type}"
                    )
                    break
                time.sleep(policy.poll_interval_seconds)
                continue
            telemetry_failure_streak = 0
            rss = 0.0
            if rss_telemetry_state == "AVAILABLE":
                try:
                    rss = rss_fn(process.pid)
                except (HostResourceSafetyFault, OSError, subprocess.SubprocessError) as exc:
                    telemetry_source = "process_group_rss"
                    telemetry_fault_type = type(exc).__name__
                    telemetry_failure_streak += 1
                    maximum_telemetry_failure_streak = max(
                        maximum_telemetry_failure_streak,
                        telemetry_failure_streak,
                    )
                    if len(telemetry_failure_prefix) < 8:
                        telemetry_failure_prefix.append(
                            {
                                "elapsed_seconds": round(elapsed, 3),
                                "source": telemetry_source,
                                "fault_type": telemetry_fault_type,
                            }
                        )
                    if isinstance(exc, PermissionError):
                        rss_permission_failure_streak += 1
                        maximum_telemetry_failure_streak = max(
                            maximum_telemetry_failure_streak,
                            rss_permission_failure_streak,
                        )
                        if (
                            rss_permission_failure_streak
                            >= MAXIMUM_CONSECUTIVE_TELEMETRY_FAILURES
                        ):
                            # RSS is supplementary on unified-memory hosts.
                            # Continue with authoritative host pressure and swap
                            # telemetry when sandbox policy denies /bin/ps.
                            rss_telemetry_state = (
                                "UNAVAILABLE_PERMISSION_HOST_PRESSURE_ACTIVE"
                            )
                    else:
                        if elapsed > policy.maximum_wall_seconds:
                            fault = "wall_limit_exceeded"
                            break
                        if (
                            telemetry_failure_streak
                            >= MAXIMUM_CONSECUTIVE_TELEMETRY_FAILURES
                        ):
                            fault = (
                                f"safety_telemetry_unavailable:{telemetry_source}:"
                                f"{telemetry_fault_type}"
                            )
                            break
                        time.sleep(policy.poll_interval_seconds)
                        continue
                else:
                    telemetry_failure_streak = 0
                    rss_permission_failure_streak = 0
            swap_growth = max(0.0, snapshot.swapouts_mib - initial.swapouts_mib)
            maximum_rss = max(maximum_rss, rss)
            inferred_unified_memory = max(
                rss,
                initial.reclaimable_available_mib
                - snapshot.reclaimable_available_mib,
            )
            maximum_inferred_unified_memory = max(
                maximum_inferred_unified_memory, inferred_unified_memory
            )
            minimum_available = min(
                minimum_available, snapshot.reclaimable_available_mib
            )
            maximum_swap_growth = max(maximum_swap_growth, swap_growth)
            observation = {
                "elapsed_seconds": round(elapsed, 3),
                "process_rss_mib": round(rss, 3),
                "inferred_unified_memory_mib": round(
                    inferred_unified_memory, 3
                ),
                "reclaimable_available_mib": round(
                    snapshot.reclaimable_available_mib, 3
                ),
                "swapout_growth_mib": round(swap_growth, 3),
            }
            if len(observation_prefix) < 64:
                observation_prefix.append(observation)
            observation_suffix.append(observation)
            if swap_growth > policy.maximum_swapout_growth_mib:
                swap_growth_breach_observations += 1
                if first_swap_growth_breach_observation is None:
                    first_swap_growth_breach_observation = observation
            if rss > policy.max_process_memory_mib:
                fault = "process_memory_limit_exceeded"
            elif (
                swap_growth > policy.maximum_swapout_growth_mib
                and policy.swapout_growth_action == "hard_stop"
            ):
                fault = "swap_growth_limit_exceeded"
            elif elapsed > policy.maximum_wall_seconds:
                fault = "wall_limit_exceeded"
            else:
                if (
                    snapshot.reclaimable_available_mib
                    < policy.minimum_available_during_run_mib
                ):
                    reserve_breach_streak += 1
                else:
                    reserve_breach_streak = 0
                maximum_reserve_breach_streak = max(
                    maximum_reserve_breach_streak, reserve_breach_streak
                )
                if (
                    reserve_breach_streak
                    >= MINIMUM_CONSECUTIVE_RESERVE_BREACHES
                ):
                    fault = "host_memory_reserve_breached"
            if fault:
                fault_observation = observation
                break
            time.sleep(policy.poll_interval_seconds)
        if fault and process.poll() is None:
            _terminate_process_group(process, policy.terminate_grace_seconds)
        returncode = int(process.wait())
        restore_interrupt_handler()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
    receipt = {
        "policy": POLICY,
        "command": [str(value) for value in command],
        "passed": returncode == 0 and not fault,
        "child_started": True,
        "terminated_by_guard": bool(fault),
        "fault": fault,
        "returncode": returncode,
        "wall_seconds": round(time.monotonic() - started, 3),
        "physical_memory_mib": round(initial.physical_memory_mib, 3),
        "initial_reclaimable_available_mib": round(
            initial.reclaimable_available_mib, 3
        ),
        "minimum_reclaimable_available_mib": round(minimum_available, 3),
        "maximum_process_rss_mib": round(maximum_rss, 3),
        "maximum_inferred_unified_memory_mib": round(
            maximum_inferred_unified_memory, 3
        ),
        "maximum_swapout_growth_mib": round(maximum_swap_growth, 3),
        "swapout_growth_action": policy.swapout_growth_action,
        "swapout_growth_threshold_exceeded": bool(
            swap_growth_breach_observations
        ),
        "swapout_growth_breach_observations": (
            swap_growth_breach_observations
        ),
        "first_swapout_growth_breach_observation": (
            first_swap_growth_breach_observation
        ),
        "reserve_breach_observations_required": MINIMUM_CONSECUTIVE_RESERVE_BREACHES,
        "maximum_consecutive_reserve_breaches": maximum_reserve_breach_streak,
        "telemetry_failure_observations_allowed": (
            MAXIMUM_CONSECUTIVE_TELEMETRY_FAILURES - 1
        ),
        "maximum_consecutive_telemetry_failures": maximum_telemetry_failure_streak,
        "telemetry_failure_prefix": telemetry_failure_prefix,
        "process_rss_telemetry_state": rss_telemetry_state,
        "limits": asdict(policy),
        "observation_prefix": observation_prefix,
        "observation_suffix": list(observation_suffix),
        "fault_observation": fault_observation,
    }
    return GuardedProcessResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        receipt=receipt,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt")
    parser.add_argument("--max-wall-seconds", type=float, default=60.0)
    parser.add_argument("--max-process-memory-mib", type=float)
    parser.add_argument("--minimum-available-before-launch-mib", type=float)
    parser.add_argument("--minimum-available-during-run-mib", type=float)
    parser.add_argument("--maximum-swapout-growth-mib", type=float)
    parser.add_argument(
        "--swapout-growth-action",
        choices=sorted(SWAPOUT_GROWTH_ACTIONS),
        default=None,
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if command:
        if not args.receipt:
            parser.error("--receipt is required when executing a guarded command")
        policy = policy_with_overrides(
            default_policy(maximum_wall_seconds=args.max_wall_seconds),
            max_process_memory_mib=args.max_process_memory_mib,
            minimum_available_before_launch_mib=(
                args.minimum_available_before_launch_mib
            ),
            minimum_available_during_run_mib=(
                args.minimum_available_during_run_mib
            ),
            maximum_swapout_growth_mib=args.maximum_swapout_growth_mib,
            swapout_growth_action=args.swapout_growth_action,
        )
        receipt_path = Path(args.receipt).expanduser()
        if not receipt_path.is_absolute():
            receipt_path = Path.cwd() / receipt_path
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = run_guarded(
                command,
                cwd=Path.cwd(),
                policy=policy,
                env={"THESEUS_GUARDED_ACCELERATOR_CHILD": "1"},
            )
            receipt = result.receipt
        except HostResourceSafetyFault as exc:
            result = None
            receipt = {
                "policy": POLICY,
                "command": [str(value) for value in command],
                "passed": False,
                "child_started": False,
                "terminated_by_guard": False,
                "fault": str(exc),
                "returncode": None,
                "limits": asdict(policy),
            }
        temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(receipt_path)
        if result is not None and result.stdout:
            print(result.stdout[-4000:], end="" if result.stdout.endswith("\n") else "\n")
        if result is not None and result.stderr:
            print(result.stderr[-4000:], end="" if result.stderr.endswith("\n") else "\n")
        print(
            json.dumps(
                {
                    "passed": receipt["passed"],
                    "fault": receipt["fault"],
                    "returncode": None if result is None else result.returncode,
                    "receipt": str(receipt_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if receipt["passed"] else 2
    snapshot = host_memory_snapshot()
    policy = default_policy(maximum_wall_seconds=args.max_wall_seconds)
    print(
        json.dumps(
            {
                "policy": POLICY,
                "snapshot": asdict(snapshot),
                "default_limits": asdict(policy),
                "preflight_passed": (
                    snapshot.reclaimable_available_mib
                    >= policy.minimum_available_before_launch_mib
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
