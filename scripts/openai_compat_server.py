"""Local OpenAI-compatible endpoint for Project Theseus.

This is an adapter, not an external inference client. It exposes the familiar
OpenAI chat/completions shape for local harnesses and routes prompts into the
grounded live/checkpoint chat shim.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "openai_compat_policy.json"
sys.path.insert(0, str(ROOT / "scripts"))
import license_manager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(POLICY_PATH.relative_to(ROOT)))
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status")
    status.add_argument("--out", default="")

    configure = sub.add_parser("configure")
    state = configure.add_mutually_exclusive_group()
    state.add_argument("--enable", action="store_true")
    state.add_argument("--disable", action="store_true")
    configure.add_argument("--host", default="")
    configure.add_argument("--port", type=int, default=0)
    configure.add_argument("--model", default="")
    configure.add_argument("--checkpoint-id", default="")
    configure.add_argument("--allow-teacher", action="store_true")
    configure.add_argument("--no-teacher", action="store_true")
    configure.add_argument("--require-token", action="store_true")
    configure.add_argument("--no-token-required", action="store_true")
    configure.add_argument("--api-token", default="")
    configure.add_argument("--out", default="")

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="")
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--out", default="")

    stop = sub.add_parser("stop")
    stop.add_argument("--out", default="")

    args = parser.parse_args()
    policy = read_json(ROOT / args.policy, {})
    if args.command == "configure":
        report = configure_endpoint(policy, args)
    elif args.command == "serve":
        return serve_endpoint(policy, args)
    elif args.command == "stop":
        report = stop_endpoint(policy)
    else:
        report = status_report(policy=policy, write_report=True)
    out = getattr(args, "out", "") or ""
    if out:
        write_json(ROOT / out, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) else 2


def configure_endpoint(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = effective_config(policy)
    if args.enable:
        cfg["enabled"] = True
    if args.disable:
        cfg["enabled"] = False
    if args.host:
        cfg["host"] = args.host
    if args.port:
        cfg["port"] = int(args.port)
    if args.model:
        cfg["model"] = args.model
    if args.checkpoint_id:
        cfg["checkpoint_id"] = args.checkpoint_id
    if args.allow_teacher:
        cfg["allow_teacher"] = True
    if args.no_teacher:
        cfg["allow_teacher"] = False
    if args.require_token:
        cfg["require_token"] = True
    if args.no_token_required:
        cfg["require_token"] = False
    if args.api_token:
        cfg["api_token"] = args.api_token
    enforce_safe_defaults(policy, cfg)
    write_json(local_config_path(policy), cfg)
    append_jsonl(events_path(policy), event("configure", public_config(cfg)))
    return status_report(policy=policy, write_report=True)


def serve_endpoint(policy: dict[str, Any], args: argparse.Namespace) -> int:
    cfg = effective_config(policy)
    if args.host:
        cfg["host"] = args.host
    if args.port:
        cfg["port"] = int(args.port)
    enforce_safe_defaults(policy, cfg)
    license_check = license_manager.check_feature("local_research", write_report=True)
    if not cfg.get("enabled"):
        report = base_status(policy, cfg, live=False, ok=False, message="endpoint_disabled", license_check=license_check)
        write_json(status_path(policy), report)
        print(json.dumps(report, indent=2))
        return 2
    if not license_check.get("allowed"):
        report = base_status(policy, cfg, live=False, ok=False, message="license_required", license_check=license_check)
        write_json(status_path(policy), report)
        print(json.dumps(report, indent=2))
        return 2

    host = str(cfg.get("host") or "127.0.0.1")
    port = int(cfg.get("port") or 8789)
    Handler.config = cfg
    Handler.policy = policy
    server = ThreadingHTTPServer((host, port), Handler)
    report = base_status(policy, cfg, live=True, ok=True, message="serving", license_check=license_check, pid=os.getpid())
    write_json(status_path(policy), report)
    append_jsonl(events_path(policy), event("serve_start", {"pid": os.getpid(), "base_url": report["base_url"]}))
    print(json.dumps(report, indent=2))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        stopped = base_status(policy, cfg, live=False, ok=True, message="stopped", license_check=license_check, pid=0)
        write_json(status_path(policy), stopped)
        append_jsonl(events_path(policy), event("serve_stop", {"pid": os.getpid()}))
    return 0


class Handler(BaseHTTPRequestHandler):
    server_version = "TheseusOpenAICompat/0.1"
    config: dict[str, Any] = {}
    policy: dict[str, Any] = {}

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler naming.
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_common_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler naming.
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/health", "/api/status"}:
            license_check = license_manager.check_feature("local_research", write_report=True)
            return self.send_json(
                base_status(
                    self.policy,
                    self.config,
                    live=True,
                    ok=True,
                    message="serving",
                    license_check=license_check,
                    pid=os.getpid(),
                )
            )
        if parsed.path == "/v1/models":
            if not self.authorized():
                return self.send_auth_error()
            return self.send_json(models_response(self.config))
        return self.send_json({"error": {"message": "not_found", "type": "invalid_request_error"}}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler naming.
        parsed = urlparse(self.path)
        if not self.authorized():
            return self.send_auth_error()
        payload = self.read_json_body()
        if parsed.path == "/v1/chat/completions":
            return self.handle_chat(payload)
        if parsed.path == "/v1/completions":
            return self.handle_completion(payload)
        return self.send_json({"error": {"message": "not_found", "type": "invalid_request_error"}}, status=404)

    def handle_chat(self, payload: dict[str, Any]) -> None:
        if not self.config.get("enabled"):
            return self.send_json({"error": {"message": "endpoint_disabled", "type": "access_error"}}, status=403)
        prompt = prompt_from_messages(payload.get("messages"))
        if not prompt:
            return self.send_json({"error": {"message": "messages_required", "type": "invalid_request_error"}}, status=400)
        model = str(payload.get("model") or self.config.get("model") or "theseus-live")
        result = local_answer(prompt, model, self.config, self.policy)
        if payload.get("stream"):
            return self.send_chat_stream(model, result)
        return self.send_json(chat_completion_response(model, result, payload))

    def handle_completion(self, payload: dict[str, Any]) -> None:
        prompt = payload.get("prompt")
        if isinstance(prompt, list):
            prompt = "\n".join(str(item) for item in prompt)
        prompt = str(prompt or "")
        if not prompt:
            return self.send_json({"error": {"message": "prompt_required", "type": "invalid_request_error"}}, status=400)
        model = str(payload.get("model") or self.config.get("model") or "theseus-live")
        result = local_answer(prompt, model, self.config, self.policy)
        return self.send_json(text_completion_response(model, result, payload))

    def send_chat_stream(self, model: str, result: dict[str, Any]) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_common_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunk = {
            "id": f"chatcmpl-theseus-{int(time.time() * 1000)}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": result.get("content", "")}, "finish_reason": None}],
        }
        final = {
            "id": chunk["id"],
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else ""
        try:
            value = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def authorized(self) -> bool:
        token = str(self.config.get("api_token") or "")
        if not self.config.get("require_token") and not token:
            return True
        auth_header = str(self.headers.get("Authorization") or "")
        prefix = "Bearer "
        supplied = auth_header[len(prefix) :] if auth_header.startswith(prefix) else auth_header
        return bool(token and supplied == token)

    def send_auth_error(self) -> None:
        self.send_json({"error": {"message": "unauthorized", "type": "authentication_error"}}, status=401)

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_common_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_common_headers(self) -> None:
        if self.config.get("cors", True):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-store")

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def local_answer(prompt: str, model: str, cfg: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    checkpoint_id = checkpoint_for_model(model, cfg)
    out = ROOT / str(get_path(policy, ["paths", "last_chat"], "reports/openai_compat_last_chat.json"))
    command = [
        sys.executable,
        "scripts/checkpoint_chat.py",
        "--checkpoint-id",
        checkpoint_id,
        "--prompt",
        prompt,
        "--out",
        str(out.relative_to(ROOT)),
    ]
    if cfg.get("allow_teacher"):
        command.append("--allow-teacher")
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=1800 if cfg.get("allow_teacher") else 180)
    except subprocess.TimeoutExpired as exc:
        runtime_ms = int((time.perf_counter() - started) * 1000)
        append_jsonl(
            events_path(policy),
            event(
                "chat_completion_timeout",
                {
                    "ok": False,
                    "model": model,
                    "checkpoint_id": checkpoint_id,
                    "runtime_ms": runtime_ms,
                    "external_inference_calls": 0,
                },
            ),
        )
        return {
            "ok": False,
            "content": f"Local Theseus checkpoint chat timed out after {runtime_ms} ms.",
            "checkpoint_id": checkpoint_id,
            "mode": "checkpoint_chat_timeout",
            "evidence": {},
            "external_inference_calls": 0,
            "teacher_used": False,
            "stderr_tail": str(exc)[-1000:],
        }
    payload = read_json(out, {})
    response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
    content = response.get("answer") or result.stdout[-4000:] or result.stderr[-1000:] or "No local answer was produced."
    ok = result.returncode == 0
    personality_context = response.get("personality_context") if isinstance(response.get("personality_context"), dict) else {}
    runtime_enforcement = payload.get("legacy_runtime_enforcement") if isinstance(payload.get("legacy_runtime_enforcement"), dict) else {}
    append_jsonl(
        events_path(policy),
        event(
            "chat_completion",
            {
                "ok": ok,
                "model": model,
                "checkpoint_id": checkpoint_id,
                "runtime_ms": int((time.perf_counter() - started) * 1000),
                "external_inference_calls": 0 if not cfg.get("allow_teacher") else get_path(payload, ["response", "teacher", "returncode"], None),
                "personality_context_status": personality_context.get("status"),
                "personality_selected_cards": get_path(personality_context, ["summary", "selected_cards"], 0),
                "runtime_enforcement_state": runtime_enforcement.get("trigger_state"),
                "runtime_ready_for_long_autonomy": runtime_enforcement.get("ready_for_long_autonomy"),
            },
        ),
    )
    return {
        "ok": ok,
        "content": str(content),
        "checkpoint_id": checkpoint_id,
        "mode": response.get("mode", "checkpoint_chat"),
        "evidence": response.get("evidence", {}),
        "personality_context": personality_context,
        "legacy_runtime_enforcement": runtime_enforcement,
        "external_inference_calls": 0,
        "teacher_used": bool(cfg.get("allow_teacher")),
        "stderr_tail": result.stderr[-1000:],
    }


def chat_completion_response(model: str, result: dict[str, Any], request_payload: dict[str, Any]) -> dict[str, Any]:
    content = str(result.get("content") or "")
    prompt_tokens = rough_tokens(json.dumps(request_payload.get("messages", [])))
    completion_tokens = rough_tokens(content)
    return {
        "id": f"chatcmpl-theseus-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": "project-theseus-local",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "theseus_context": {
            "checkpoint_id": result.get("checkpoint_id"),
            "mode": result.get("mode"),
            "personality_context": result.get("personality_context", {}),
            "legacy_runtime_enforcement": result.get("legacy_runtime_enforcement", {}),
        },
    }


def text_completion_response(model: str, result: dict[str, Any], request_payload: dict[str, Any]) -> dict[str, Any]:
    content = str(result.get("content") or "")
    prompt_tokens = rough_tokens(str(request_payload.get("prompt") or ""))
    completion_tokens = rough_tokens(content)
    return {
        "id": f"cmpl-theseus-{int(time.time() * 1000)}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"text": content, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "theseus_context": {
            "checkpoint_id": result.get("checkpoint_id"),
            "mode": result.get("mode"),
            "personality_context": result.get("personality_context", {}),
            "legacy_runtime_enforcement": result.get("legacy_runtime_enforcement", {}),
        },
    }


def models_response(cfg: dict[str, Any]) -> dict[str, Any]:
    model = str(cfg.get("model") or "theseus-live")
    model_ids = [model]
    if model != "theseus-live":
        model_ids.append("theseus-live")
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model", "created": 0, "owned_by": "project-theseus"} for model_id in model_ids],
    }


def prompt_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts = []
    for row in messages:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "user")
        content = row.get("content")
        if isinstance(content, list):
            text = " ".join(str(item.get("text") if isinstance(item, dict) else item) for item in content)
        else:
            text = str(content or "")
        if text:
            parts.append(f"{role}: {text}")
    return "\n".join(parts)


def checkpoint_for_model(model: str, cfg: dict[str, Any]) -> str:
    if model.startswith("theseus-checkpoint:"):
        return model.split(":", 1)[1] or "live"
    return str(cfg.get("checkpoint_id") or "live")


def status_report(*, policy: dict[str, Any] | None = None, write_report: bool = False) -> dict[str, Any]:
    policy = policy or read_json(POLICY_PATH, {})
    cfg = effective_config(policy)
    license_check = license_manager.check_feature("local_research", write_report=True)
    health = probe_health(str(cfg.get("host") or "127.0.0.1"), int(cfg.get("port") or 8789))
    live = bool(health)
    live_pid = int(health.get("pid") or read_json(status_path(policy), {}).get("pid") or 0) if live else 0
    report = base_status(policy, cfg, live=live, ok=True, message="live" if live else "stopped", license_check=license_check, pid=live_pid)
    if write_report:
        write_json(status_path(policy), report)
    return report


def stop_endpoint(policy: dict[str, Any]) -> dict[str, Any]:
    status = read_json(status_path(policy), {})
    status_pid = int(status.get("pid") or 0)
    pids = sorted(set([pid for pid in [status_pid, *matching_server_pids()] if pid and pid != os.getpid()]))
    stopped_pids: list[int] = []
    for pid in pids:
        try:
            terminate_pid(pid)
            stopped_pids.append(pid)
        except OSError:
            continue
    cfg = effective_config(policy)
    report = base_status(
        policy,
        cfg,
        live=False,
        ok=True,
        message="stop_requested" if stopped_pids else "not_running",
        license_check=license_manager.check_feature("local_research", write_report=True),
        pid=0,
    )
    write_json(status_path(policy), report)
    append_jsonl(events_path(policy), event("stop", {"status_pid": status_pid, "stopped_pids": stopped_pids}))
    return report


def base_status(
    policy: dict[str, Any],
    cfg: dict[str, Any],
    *,
    live: bool,
    ok: bool,
    message: str,
    license_check: dict[str, Any],
    pid: int | None = None,
) -> dict[str, Any]:
    host = str(cfg.get("host") or "127.0.0.1")
    port = int(cfg.get("port") or 8789)
    previous_pid = int(read_json(status_path(policy), {}).get("pid") or 0)
    status_pid = int(pid if pid is not None else (previous_pid if live else 0))
    return {
        "ok": ok,
        "policy": "project_theseus_openai_compat_status_v0",
        "created_utc": now(),
        "enabled": bool(cfg.get("enabled")),
        "live": bool(live),
        "message": message,
        "pid": status_pid,
        "host": host,
        "port": port,
        "base_url": f"http://{host}:{port}/v1",
        "chat_completions_url": f"http://{host}:{port}/v1/chat/completions",
        "models_url": f"http://{host}:{port}/v1/models",
        "model": cfg.get("model"),
        "checkpoint_id": cfg.get("checkpoint_id"),
        "accept_any_model": bool(cfg.get("accept_any_model", True)),
        "require_token": bool(cfg.get("require_token")),
        "token_configured": bool(cfg.get("api_token")),
        "allow_teacher": bool(cfg.get("allow_teacher")),
        "external_inference_calls": 0,
        "license": {
            "allowed": bool(license_check.get("allowed")),
            "tier": get_path(license_check, ["entitlement", "tier"], None),
            "source": get_path(license_check, ["entitlement", "source"], None),
            "next_action": license_check.get("next_action"),
        },
        "usage_hint": {
            "base_url": f"http://{host}:{port}/v1",
            "api_key": "any value" if not cfg.get("require_token") else "configured local token",
            "model": cfg.get("model") or "theseus-live",
        },
    }


def enforce_safe_defaults(policy: dict[str, Any], cfg: dict[str, Any]) -> None:
    host = str(cfg.get("host") or "127.0.0.1")
    if get_path(policy, ["security", "loopback_only_by_default"], True) and host in {"", "0.0.0.0", "::"}:
        if not cfg.get("allow_lan"):
            cfg["host"] = "127.0.0.1"
    if not is_loopback(str(cfg.get("host") or "127.0.0.1")) and get_path(policy, ["security", "require_token_for_non_loopback"], True):
        cfg["require_token"] = True
    if get_path(policy, ["security", "never_call_external_inference"], True):
        cfg["allow_teacher"] = False
    if get_path(policy, ["security", "teacher_disabled_by_default"], True) and "allow_teacher" not in cfg:
        cfg["allow_teacher"] = False


def effective_config(policy: dict[str, Any]) -> dict[str, Any]:
    cfg = {}
    defaults = policy.get("defaults") if isinstance(policy.get("defaults"), dict) else {}
    cfg.update(defaults)
    local = read_json(local_config_path(policy), {})
    if isinstance(local, dict):
        cfg.update(local)
    return cfg


def public_config(cfg: dict[str, Any]) -> dict[str, Any]:
    clean = dict(cfg)
    clean["api_token"] = "***" if clean.get("api_token") else ""
    return clean


def port_live(host: str, port: int) -> bool:
    return bool(probe_health(host, port))


def probe_health(host: str, port: int) -> dict[str, Any]:
    try:
        with urlrequest.urlopen(f"http://{host}:{port}/health", timeout=1) as response:
            if response.status != 200:
                return {}
            payload = json.loads(response.read().decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
    except (OSError, URLError, ValueError):
        return {}


def matching_server_pids() -> list[int]:
    if os.name == "nt":
        script = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -match 'openai_compat_server\\.py serve' } | "
            "ForEach-Object { $_.ProcessId }"
        )
        result = subprocess.run(["powershell", "-NoProfile", "-Command", script], text=True, capture_output=True, timeout=5)
        return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    result = subprocess.run(["pgrep", "-f", "openai_compat_server.py serve"], text=True, capture_output=True, timeout=5)
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


def terminate_pid(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], text=True, capture_output=True, timeout=10)
        return
    os.kill(pid, signal.SIGTERM)


def is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def rough_tokens(text: str) -> int:
    return max(1, len(text.split()))


def local_config_path(policy: dict[str, Any]) -> Path:
    return ROOT / str(get_path(policy, ["paths", "local_config"], "configs/openai_compat.local.json"))


def status_path(policy: dict[str, Any]) -> Path:
    return ROOT / str(get_path(policy, ["paths", "status"], "reports/openai_compat_status.json"))


def events_path(policy: dict[str, Any]) -> Path:
    return ROOT / str(get_path(policy, ["paths", "events"], "reports/openai_compat_events.jsonl"))


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def event(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"created_utc": now(), "kind": kind, **payload}


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
