"""Worker slot accounting for the Project Theseus Hive node."""

from __future__ import annotations

import threading
from typing import Any

from hive_node_common import get_path
from hive_node_resources import probe_resources, resource_slots


SLOTS_LOCK = threading.Lock()
SLOTS: dict[str, dict[str, Any]] = {}


def init_slot_state(status: dict[str, Any]) -> None:
    with SLOTS_LOCK:
        SLOTS.clear()
        for slot in status.get("slots") or []:
            SLOTS[str(slot["slot_id"])] = {**slot, "running": 0}


def slots_snapshot() -> list[dict[str, Any]]:
    with SLOTS_LOCK:
        return [slot_with_availability(slot) for slot in SLOTS.values()]


def slot_with_availability(slot: dict[str, Any]) -> dict[str, Any]:
    capacity = int(slot.get("capacity") or 0)
    running = int(slot.get("running") or 0)
    return {
        **slot,
        "running": running,
        "available": capacity <= 0 or running < capacity,
    }


def local_task_support(policy: dict[str, Any], kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    accepted = set(slot_types_for_task(policy, kind, payload))
    with SLOTS_LOCK:
        candidates = [slot_with_availability(slot) for slot in SLOTS.values()]
    if not candidates:
        candidates = [slot_with_availability(slot) for slot in resource_slots(probe_resources(policy), policy)]
    matching = []
    for slot in candidates:
        slot_type = str(slot.get("slot_type") or "")
        task_kinds = set(str(item) for item in (slot.get("task_kinds") or []))
        if slot_type in accepted and (not task_kinds or kind in task_kinds):
            matching.append(slot)
    if matching:
        return {"ok": True, "slot_types": sorted(accepted), "matching_slots": matching}
    return {
        "ok": False,
        "reason": "no_available_slot_for_task",
        "slot_types": sorted(accepted),
        "matching_slots": matching,
        "slots": candidates,
    }


def acquire_slot(policy: dict[str, Any], kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    accepted = set(slot_types_for_task(policy, kind, payload))
    with SLOTS_LOCK:
        for slot_id, slot in SLOTS.items():
            task_kinds = set(str(item) for item in (slot.get("task_kinds") or []))
            if str(slot.get("slot_type") or "") not in accepted:
                continue
            if task_kinds and kind not in task_kinds:
                continue
            if int(slot.get("capacity") or 0) <= 0 or int(slot.get("running") or 0) < int(slot.get("capacity") or 0):
                slot["running"] = int(slot.get("running") or 0) + 1
                return {"slot_id": slot_id, **slot}
    return None


def release_slot(slot_id: str) -> None:
    with SLOTS_LOCK:
        if slot_id in SLOTS:
            SLOTS[slot_id]["running"] = max(0, int(SLOTS[slot_id].get("running") or 0) - 1)


def slot_types_for_task(policy: dict[str, Any], kind: str, payload: dict[str, Any]) -> list[str]:
    explicit = payload.get("slot_types") if isinstance(payload.get("slot_types"), list) else []
    if explicit:
        return [str(item) for item in explicit]
    by_kind = get_path(policy, ["resource_slots", "slot_types_by_task_kind", kind], [])
    if isinstance(by_kind, list) and by_kind:
        return [str(item) for item in by_kind]
    if kind.startswith("cuda_"):
        return ["cuda"]
    if kind.startswith("mlx_"):
        return ["mlx_apple", "mlx_cuda"]
    return ["cpu"]
