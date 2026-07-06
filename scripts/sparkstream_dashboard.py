"""Live SparkStream dashboard server.

This uses only the Python standard library so it can run before the project has
a web stack. It serves a small dashboard, streams status updates, and can launch
bounded local SparkStream commands.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import license_manager  # noqa: E402
import compute_market  # noqa: E402
import openai_compat_server  # noqa: E402
import update_manager  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "dashboard"
REPORTS = ROOT / "reports"
ACTIVE_JOBS: dict[str, dict[str, Any]] = {}
ACTIVE_LOCK = threading.Lock()
LONG_RUN_PROFILES = {"candidate", "seed_sweep"}
SINGLETON_JOB_NEEDLES = {
    "sparkstream_daemon": ["sparkstream_daemon.py"],
    "hive_node": ["hive_node.py daemon"],
    "hive_relay": ["hive_relay.py"],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SparkStream dashboard: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


class Handler(BaseHTTPRequestHandler):
    server_version = "SparkStreamDashboard/0.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler naming.
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self.serve_file(DASHBOARD_DIR / "index.html", "text/html; charset=utf-8")
        if parsed.path == "/styles.css":
            return self.serve_file(DASHBOARD_DIR / "styles.css", "text/css; charset=utf-8")
        if parsed.path == "/app.js":
            return self.serve_file(DASHBOARD_DIR / "app.js", "text/javascript; charset=utf-8")
        if parsed.path == "/manifest.webmanifest":
            return self.serve_file(DASHBOARD_DIR / "manifest.webmanifest", "application/manifest+json; charset=utf-8")
        if parsed.path == "/service-worker.js":
            return self.serve_file(DASHBOARD_DIR / "service-worker.js", "text/javascript; charset=utf-8")
        if parsed.path == "/api/health":
            return self.send_json(build_health())
        if parsed.path == "/api/status":
            return self.send_json(build_status())
        if parsed.path == "/api/events":
            return self.serve_events()
        return self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler naming.
        parsed = urlparse(self.path)
        payload = self.read_json_body()
        if parsed.path == "/api/control":
            return self.send_json(handle_control(payload))
        if parsed.path == "/api/license/status":
            return self.send_json(license_manager.status_report(write_report=True))
        if parsed.path == "/api/license/register":
            args = argparse.Namespace(
                name=str(payload.get("name") or ""),
                email=str(payload.get("email") or ""),
                organization=str(payload.get("organization") or ""),
                usage=str(payload.get("usage") or "personal_homelab"),
                seats=int(payload.get("seats") or 1),
                commercial=bool(payload.get("commercial")),
                accept_terms=bool(payload.get("accept_terms")),
            )
            return self.send_json(license_manager.register_install(license_manager.read_json(license_manager.POLICY_PATH, {}), args))
        if parsed.path == "/api/license/request":
            features = payload.get("features") if isinstance(payload.get("features"), list) else []
            feature = str(payload.get("feature") or "")
            if feature:
                features.append(feature)
            return self.send_json(license_manager.license_request(license_manager.read_json(license_manager.POLICY_PATH, {}), [str(item) for item in features]))
        if parsed.path == "/api/license/import":
            raw = payload.get("license_json", "")
            if isinstance(raw, dict):
                raw = json.dumps(raw)
            return self.send_json(
                license_manager.import_license(
                    license_manager.read_json(license_manager.POLICY_PATH, {}),
                    file_path=str(payload.get("file") or ""),
                    raw=str(raw or ""),
                )
            )
        if parsed.path == "/api/openai/status":
            return self.send_json(openai_compat_server.status_report(write_report=True))
        if parsed.path == "/api/openai/configure":
            args = argparse.Namespace(
                enable=bool(payload.get("enabled")),
                disable=payload.get("enabled") is False,
                host=str(payload.get("host") or ""),
                port=int(payload.get("port") or 0),
                model=str(payload.get("model") or ""),
                checkpoint_id=str(payload.get("checkpoint_id") or ""),
                allow_teacher=False,
                no_teacher=True,
                require_token=bool(payload.get("require_token")),
                no_token_required=payload.get("require_token") is False,
                api_token=str(payload.get("api_token") or ""),
            )
            return self.send_json(openai_compat_server.configure_endpoint(openai_compat_server.read_json(openai_compat_server.POLICY_PATH, {}), args))
        if parsed.path == "/api/openai/start":
            policy = openai_compat_server.read_json(openai_compat_server.POLICY_PATH, {})
            cfg = openai_compat_server.effective_config(policy)
            cfg["enabled"] = True
            if payload.get("host"):
                cfg["host"] = str(payload.get("host"))
            if payload.get("port"):
                cfg["port"] = int(payload.get("port"))
            if payload.get("model"):
                cfg["model"] = str(payload.get("model"))
            if payload.get("checkpoint_id"):
                cfg["checkpoint_id"] = str(payload.get("checkpoint_id"))
            cfg["allow_teacher"] = False
            openai_compat_server.enforce_safe_defaults(policy, cfg)
            openai_compat_server.write_json(openai_compat_server.local_config_path(policy), cfg)
            command = [
                sys.executable,
                "scripts/openai_compat_server.py",
                "serve",
            ]
            return self.send_json(start_job("openai_compat_server", command))
        if parsed.path == "/api/openai/stop":
            return self.send_json(openai_compat_server.stop_endpoint(openai_compat_server.read_json(openai_compat_server.POLICY_PATH, {})))
        if parsed.path == "/api/updates/status":
            return self.send_json(update_manager.status_report(write_report=True))
        if parsed.path == "/api/updates/check":
            args = argparse.Namespace(
                catalog_url=str(payload.get("catalog_url") or ""),
                update_id=str(payload.get("update_id") or ""),
                apply=bool(payload.get("apply")),
                if_enabled_on_start=bool(payload.get("if_enabled_on_start")),
                respect_interval=bool(payload.get("respect_interval")),
            )
            return self.send_json(update_manager.check_for_updates(update_manager.read_json(update_manager.POLICY_PATH, {}), args))
        if parsed.path == "/api/updates/configure":
            args = argparse.Namespace(
                mode=str(payload.get("mode") or ""),
                channel=str(payload.get("channel") or ""),
                track=str(payload.get("track") or ""),
                catalog_url=str(payload.get("catalog_url") or ""),
                check_on_start=bool(payload.get("check_on_start")),
                no_check_on_start=bool(payload.get("no_check_on_start")),
                auto_install_soft=bool(payload.get("auto_install_soft")),
                no_auto_install_soft=bool(payload.get("no_auto_install_soft")),
                auto_install_hard=bool(payload.get("auto_install_hard")),
                no_auto_install_hard=bool(payload.get("no_auto_install_hard")),
                allow_prerelease=bool(payload.get("allow_prerelease")),
                no_allow_prerelease=bool(payload.get("no_allow_prerelease")),
            )
            return self.send_json(update_manager.configure_client(update_manager.read_json(update_manager.POLICY_PATH, {}), args))
        if parsed.path == "/api/updates/catalog":
            return self.send_json(update_manager.public_catalog(update_manager.read_json(update_manager.POLICY_PATH, {})))
        if parsed.path == "/api/updates/create":
            args = argparse.Namespace(
                checkpoint_id=str(payload.get("checkpoint_id") or ""),
                if_promoted=bool(payload.get("if_promoted", True)),
            )
            return self.send_json(update_manager.create_offer(update_manager.read_json(update_manager.POLICY_PATH, {}), args))
        if parsed.path == "/api/updates/apply":
            args = argparse.Namespace(
                mode=str(payload.get("mode") or "auto"),
                execute=bool(payload.get("execute")),
                allow_hard=bool(payload.get("allow_hard")),
                restart=bool(payload.get("restart")),
                offer=str(payload.get("offer") or ""),
            )
            if args.mode not in {"auto", "soft", "hard"}:
                return self.send_json({"ok": False, "error": "invalid_update_mode"}, status=400)
            return self.send_json(update_manager.apply_update(update_manager.read_json(update_manager.POLICY_PATH, {}), args))
        if parsed.path == "/api/market/status":
            return self.send_json(compute_market.status_report(write_report=True))
        if parsed.path == "/api/market/quote":
            task_kind = str(payload.get("task_kind") or "")
            if not task_kind:
                return self.send_json({"ok": False, "error": "task_kind_required"}, status=400)
            task_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
            provider = payload.get("provider_node") if isinstance(payload.get("provider_node"), dict) else {}
            return self.send_json(compute_market.quote_task(task_kind, task_payload, provider, write_report=True))
        if parsed.path == "/api/market/settle":
            return self.send_json(
                compute_market.settle_worker_ledger(
                    ROOT / "reports" / "hive_worker_chunk_ledger.jsonl",
                    limit=int(payload.get("limit") or 50),
                    write_report=True,
                )
            )
        if parsed.path == "/api/viea/action-executor/run":
            command = [
                sys.executable,
                "scripts/viea_action_executor.py",
                "--execute",
                "--resume",
                "--max-actions",
                str(int(payload.get("max_actions") or 3)),
                "--max-steps",
                str(int(payload.get("max_steps") or 8)),
                "--timeout-seconds",
                str(int(payload.get("timeout_seconds") or 7200)),
                "--out",
                "reports/viea_action_executor.json",
                "--markdown-out",
                "reports/viea_action_executor.md",
            ]
            if payload.get("allow_teacher"):
                command.append("--allow-teacher")
            return self.send_json(start_job("viea_action_executor", command))
        if parsed.path == "/api/viea/action-executor/status":
            return self.send_json(run_now([
                sys.executable,
                "scripts/viea_action_executor.py",
                "--status",
                "--out",
                "reports/viea_action_executor.json",
                "--markdown-out",
                "reports/viea_action_executor.md",
            ]))
        if parsed.path == "/api/viea/action-executor/pause":
            return self.send_json(run_now([
                sys.executable,
                "scripts/viea_action_executor.py",
                "--pause",
                "--status",
                "--out",
                "reports/viea_action_executor.json",
                "--markdown-out",
                "reports/viea_action_executor.md",
            ]))
        if parsed.path == "/api/viea/action-executor/resume":
            return self.send_json(run_now([
                sys.executable,
                "scripts/viea_action_executor.py",
                "--resume-queue",
                "--status",
                "--out",
                "reports/viea_action_executor.json",
                "--markdown-out",
                "reports/viea_action_executor.md",
            ]))
        if parsed.path == "/api/viea/action-executor/block":
            action_id = str(payload.get("action_id") or "")
            if not action_id:
                return self.send_json({"ok": False, "error": "action_id_required"}, status=400)
            return self.send_json(run_now([
                sys.executable,
                "scripts/viea_action_executor.py",
                "--mark-blocked",
                action_id,
                "--reason",
                str(payload.get("reason") or "dashboard_blocked"),
                "--status",
                "--out",
                "reports/viea_action_executor.json",
                "--markdown-out",
                "reports/viea_action_executor.md",
            ]))
        if parsed.path == "/api/viea/teacher/request":
            command = [
                sys.executable,
                "scripts/teacher_architect_experiment_runner.py",
                "--execute",
                "--max-experiments",
                str(int(payload.get("max_experiments") or 1)),
                "--max-steps",
                str(int(payload.get("max_steps") or 2)),
                "--timeout-seconds",
                str(int(payload.get("timeout_seconds") or 7200)),
                "--out",
                "reports/teacher_architect_experiment_runner.json",
                "--markdown-out",
                "reports/teacher_architect_experiment_runner.md",
            ]
            if payload.get("allow_teacher"):
                command.append("--allow-teacher")
            return self.send_json(start_job("viea_teacher_architect_runner", command))
        if parsed.path == "/api/viea/broad-calibration/run":
            return self.send_json(start_job("viea_broad_calibration", [
                sys.executable,
                "scripts/broad_transfer_matrix.py",
                "--min-public-tasks",
                str(int(payload.get("min_public_tasks") or 32)),
                "--out",
                "reports/broad_transfer_matrix.json",
                "--markdown-out",
                "reports/broad_transfer_matrix.md",
            ]))
        if parsed.path == "/api/viea/repo-repair/refresh":
            return self.send_json(start_job("viea_repo_repair_refresh", [
                sys.executable,
                "scripts/viea_repo_repair_learner.py",
                "--out",
                "reports/viea_repo_repair_learner.json",
                "--markdown-out",
                "reports/viea_repo_repair_learner.md",
            ]))
        if parsed.path == "/api/viea/symliquid/refresh":
            return self.send_json(run_now([
                sys.executable,
                "scripts/symliquid_state_engine.py",
                "--out",
                "reports/symliquid_state_engine.json",
                "--markdown-out",
                "reports/symliquid_state_engine.md",
            ]))
        if parsed.path == "/api/readiness/run":
            profile = str(payload.get("profile") or "inner_loop")
            command = [
                sys.executable,
                "scripts/arm_lifecycle_manager.py",
                "--out",
                "reports/arm_lifecycle_governance.json",
            ]
            arm_result = run_now(command)
            readiness_command = [
                sys.executable,
                "scripts/autonomy_launch_readiness.py",
                "--profile",
                profile,
                "--out",
                "reports/autonomy_launch_readiness.json",
            ]
            if payload.get("require_teacher_cli"):
                readiness_command.append("--require-teacher-cli")
            readiness_result = run_now(readiness_command)
            return self.send_json(
                {
                    "ok": arm_result.get("ok") and readiness_result.get("ok"),
                    "arm_lifecycle": arm_result,
                    "launch_readiness": readiness_result,
                }
            )
        if parsed.path == "/api/checkpoints/create":
            kind = str(payload.get("kind") or "auto")
            if kind not in {"auto", "major", "minor"}:
                return self.send_json({"ok": False, "error": "invalid_checkpoint_kind"}, status=400)
            return self.send_json(start_job("checkpoint_create", [
                sys.executable,
                "scripts/checkpoint_registry.py",
                "create",
                "--kind",
                kind,
                "--label",
                str(payload.get("label") or "dashboard_checkpoint"),
                "--reason",
                str(payload.get("reason") or "dashboard"),
                "--profile",
                str(payload.get("profile") or "manual"),
                "--status",
                str(payload.get("status") or "recorded"),
            ]))
        if parsed.path == "/api/checkpoints/compare":
            a = str(payload.get("a") or "")
            b = str(payload.get("b") or "")
            if not a or not b:
                return self.send_json({"ok": False, "error": "a_and_b_required"}, status=400)
            return self.send_json(run_now([
                sys.executable,
                "scripts/checkpoint_registry.py",
                "compare",
                "--a",
                a,
                "--b",
                b,
            ]))
        if parsed.path == "/api/checkpoints/materialize":
            checkpoint_id = str(payload.get("checkpoint_id") or "")
            if not checkpoint_id:
                return self.send_json({"ok": False, "error": "checkpoint_id_required"}, status=400)
            command = [
                sys.executable,
                "scripts/checkpoint_registry.py",
                "materialize",
                "--id",
                checkpoint_id,
                "--out",
                str(Path("checkpoints") / "materialized" / checkpoint_id),
                "--report-out",
                "reports/checkpoint_materialize_last.json",
            ]
            if payload.get("force"):
                command.append("--force")
            return self.send_json(run_now(command))
        if parsed.path == "/api/checkpoints/backup":
            command = [
                sys.executable,
                "scripts/checkpoint_backup_manager.py",
                "--if-promoted",
                "--provider",
                str(payload.get("provider") or "all"),
                "--out",
                "reports/checkpoint_backup_last.json",
            ]
            checkpoint_id = str(payload.get("checkpoint_id") or "")
            if checkpoint_id:
                command.extend(["--checkpoint-id", checkpoint_id])
            if payload.get("execute"):
                command.append("--execute")
            return self.send_json(start_job("checkpoint_backup", command))
        if parsed.path == "/api/benchmarks/add":
            url = str(payload.get("url") or "")
            if not url:
                return self.send_json({"ok": False, "error": "url_required"}, status=400)
            if payload.get("allow_network_fetch") and not payload.get("confirm_external_fetch"):
                return self.send_json(
                    {
                        "ok": False,
                        "error": "external_fetch_confirmation_required",
                        "message": "Network benchmark/data fetches need explicit confirmation.",
                    },
                    status=400,
                )
            command = [
                sys.executable,
                "scripts/benchmark_seeker.py",
                "--add-url",
                url,
                "--name",
                str(payload.get("name") or url),
                "--notes",
                str(payload.get("notes") or "dashboard_request"),
                "--out",
                "reports/benchmark_seeker_registry.json",
            ]
            if payload.get("allow_network_fetch"):
                command.extend(["--fetch-url", url, "--allow-network-fetch"])
            return self.send_json(start_job("benchmark_add", command))
        if parsed.path == "/api/benchmarks/discover":
            query = str(payload.get("query") or "")
            if not query:
                return self.send_json({"ok": False, "error": "query_required"}, status=400)
            if not payload.get("allow_network_fetch") or not payload.get("confirm_external_fetch"):
                return self.send_json(
                    {
                        "ok": False,
                        "error": "network_discovery_confirmation_required",
                        "message": "Web discovery needs the network fetch switch and explicit confirmation.",
                    },
                    status=400,
                )
            return self.send_json(start_job("benchmark_discovery", [
                sys.executable,
                "scripts/benchmark_seeker.py",
                "--discover-query",
                query,
                "--discover-limit",
                str(payload.get("limit") or 10),
                "--allow-network-discovery",
                "--out",
                "reports/benchmark_seeker_registry.json",
            ]))
        if parsed.path == "/api/teacher/ask":
            prompt = str(payload.get("prompt") or "")
            reason = str(payload.get("reason") or "user_requested_benchmark")
            if not prompt:
                return self.send_json({"ok": False, "error": "prompt_required"}, status=400)
            command = [
                sys.executable,
                "scripts/teacher_oracle.py",
                "--reason",
                reason,
                "--mode",
                "proposal",
                "--prompt",
                prompt,
                "--local-evidence",
                "reports/autonomy_watchdog.json",
                "reports/learning_scoreboard.json",
                "reports/broad_transfer_matrix.json",
                "reports/architecture_guidance_loop.json",
                "--out",
                "reports/teacher_oracle_last.json",
            ]
            if payload.get("allow_teacher"):
                command.append("--allow-teacher")
            else:
                command.append("--queue-only")
            return self.send_json(start_job("teacher_request", command))
        if parsed.path == "/api/chat/checkpoint":
            prompt = str(payload.get("prompt") or "")
            checkpoint_id = str(payload.get("checkpoint_id") or "live")
            if not prompt:
                return self.send_json({"ok": False, "error": "prompt_required"}, status=400)
            command = [
                sys.executable,
                "scripts/checkpoint_chat.py",
                "--checkpoint-id",
                checkpoint_id,
                "--prompt",
                prompt,
                "--out",
                "reports/checkpoint_chat_last.json",
            ]
            if payload.get("allow_teacher"):
                command.append("--allow-teacher")
            return self.send_json(run_now(command, timeout=1800 if payload.get("allow_teacher") else 180))
        if parsed.path == "/api/goals/run":
            goal = str(payload.get("goal") or "")
            if not goal:
                return self.send_json({"ok": False, "error": "goal_required"}, status=400)
            profile = str(payload.get("profile") or "inner_loop")
            command = [
                sys.executable,
                "scripts/autonomous_goal_runner.py",
                "--goal",
                goal,
                "--profile",
                profile,
                "--out",
                "reports/autonomous_goal_last.json",
            ]
            if payload.get("execute"):
                command.append("--execute")
            if payload.get("allow_teacher"):
                command.append("--allow-teacher")
            if payload.get("allow_network_fetch"):
                if not payload.get("confirm_external_fetch"):
                    return self.send_json(
                        {
                            "ok": False,
                            "error": "external_fetch_confirmation_required",
                            "message": "Network goal execution needs explicit confirmation.",
                        },
                        status=400,
                    )
                command.append("--allow-network-fetch")
            return self.send_json(start_job("autonomous_goal", command))
        if parsed.path == "/api/rl/discover":
            query = str(payload.get("query") or "")
            if not query:
                return self.send_json({"ok": False, "error": "query_required"}, status=400)
            if not payload.get("allow_network_fetch") or not payload.get("confirm_external_fetch"):
                return self.send_json(
                    {
                        "ok": False,
                        "error": "network_discovery_confirmation_required",
                        "message": "RL source discovery needs the network fetch switch and explicit confirmation.",
                    },
                    status=400,
                )
            return self.send_json(start_job("rl_discovery", [
                sys.executable,
                "scripts/rl_benchmark_registry.py",
                "--refresh-local",
                "--allow-network-discovery",
                "--discover-query",
                query,
                "--discover-limit",
                str(payload.get("limit") or 10),
                "--out",
                "reports/rl_benchmark_registry.json",
            ]))
        if parsed.path == "/api/sources/catalog":
            command = [
                sys.executable,
                "scripts/online_source_catalog.py",
                "--catalog",
                "configs/online_source_catalog.json",
                "--out",
                "reports/online_source_catalog_report.json",
            ]
            if payload.get("import_sources"):
                if not payload.get("allow_network_fetch") or not payload.get("confirm_external_fetch"):
                    return self.send_json(
                        {
                            "ok": False,
                            "error": "external_fetch_confirmation_required",
                            "message": "Catalog imports need network fetch enabled and explicit confirmation.",
                        },
                        status=400,
                    )
                command.extend(
                    [
                        "--allow-network-fetch",
                        "--import-sources",
                        "--max-imports",
                        str(payload.get("max_imports") or 8),
                    ]
                )
            return self.send_json(start_job("online_source_catalog", command))
        if parsed.path == "/api/hive/probe":
            probe = run_now([
                sys.executable,
                "scripts/hive_node.py",
                "probe",
                "--out",
                "reports/hive_status.json",
                "--peers-out",
                "reports/hive_peers.json",
            ])
            scheduler = run_now([
                sys.executable,
                "scripts/hive_scheduler.py",
                "--out",
                "reports/hive_scheduler.json",
            ])
            return self.send_json({"ok": probe.get("ok") and scheduler.get("ok"), "probe": probe, "scheduler": scheduler})
        if parsed.path == "/api/hive/start":
            command = [
                sys.executable,
                "scripts/hive_node.py",
                "daemon",
            ]
            port = str(payload.get("port") or "")
            if port:
                command.extend(["--port", port])
            if payload.get("no_discovery"):
                command.append("--no-discovery")
            if payload.get("no_worker"):
                command.append("--no-worker")
            return self.send_json(start_job("hive_node", command))
        if parsed.path == "/api/hive/relay/start":
            command = [
                sys.executable,
                "scripts/hive_relay.py",
            ]
            port = str(payload.get("port") or "")
            if port:
                command.extend(["--port", port])
            return self.send_json(start_job("hive_relay", command))
        if parsed.path == "/api/hive/schedule":
            command = [
                sys.executable,
                "scripts/hive_scheduler.py",
                "--out",
                "reports/hive_scheduler.json",
            ]
            if payload.get("execute"):
                command.append("--execute")
            if payload.get("probe_peers"):
                command.append("--probe-peers")
            if payload.get("worker_chunks"):
                command.append("--worker-chunks")
            return self.send_json(start_job("hive_scheduler", command))
        if parsed.path == "/api/hive/operator-os/refresh":
            return self.send_json(start_job("hive_operator_os", [
                sys.executable,
                "scripts/hive_operator_os.py",
                "--config",
                "configs/hive_operator_os.json",
                "--db",
                "reports/hive_work_board.sqlite",
                "--out",
                "reports/hive_operator_os.json",
                "--markdown-out",
                "reports/hive_operator_os.md",
            ]))
        if parsed.path == "/api/hive/work-board/run":
            command = [
                sys.executable,
                "scripts/hive_work_board_executor.py",
                "--execute",
                "--resume",
                "--max-tasks",
                str(int(payload.get("max_tasks") or 1)),
                "--max-steps",
                str(int(payload.get("max_steps") or 1)),
                "--timeout-seconds",
                str(int(payload.get("timeout_seconds") or 21600)),
                "--out",
                "reports/hive_work_board_executor.json",
                "--markdown-out",
                "reports/hive_work_board_executor.md",
            ]
            if payload.get("allow_teacher"):
                command.append("--allow-teacher")
            return self.send_json(start_job("hive_work_board_executor", command))
        if parsed.path == "/api/hive/command":
            command_text = str(payload.get("command") or "").strip()
            if not command_text:
                return self.send_json({"ok": False, "error": "command_required"}, status=400)
            command = [
                sys.executable,
                "scripts/hive_work_board_executor.py",
                "--command-text",
                command_text,
                "--source-channel",
                str(payload.get("source_channel") or "dashboard"),
                "--max-tasks",
                str(int(payload.get("max_tasks") or 1)),
                "--max-steps",
                str(int(payload.get("max_steps") or 1)),
                "--timeout-seconds",
                str(int(payload.get("timeout_seconds") or 21600)),
                "--out",
                "reports/hive_work_board_executor.json",
                "--markdown-out",
                "reports/hive_work_board_executor.md",
            ]
            if payload.get("execute", True):
                command.append("--execute")
            if payload.get("enqueue_only"):
                command.append("--enqueue-only")
            if payload.get("allow_teacher"):
                command.append("--allow-teacher")
            return self.send_json(start_job("hive_live_command", command))
        if parsed.path == "/api/hive/high-transfer/schedule":
            return self.send_json(start_job("high_transfer_curriculum_scheduler", [
                sys.executable,
                "scripts/high_transfer_curriculum_scheduler.py",
                "--out",
                "reports/high_transfer_curriculum_scheduler.json",
                "--markdown-out",
                "reports/high_transfer_curriculum_scheduler.md",
                "--tasks-out",
                "reports/high_transfer_curriculum_tasks.jsonl",
            ]))
        if parsed.path == "/api/hive/task":
            kind = str(payload.get("kind") or "")
            if not kind:
                return self.send_json({"ok": False, "error": "kind_required"}, status=400)
            task_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
            peer_url = str(payload.get("peer_url") or "").strip()
            if peer_url:
                command = [
                    sys.executable,
                    "scripts/hive_node.py",
                    "submit",
                    "--peer-url",
                    peer_url,
                    "--kind",
                    kind,
                    "--payload-json",
                    json.dumps(task_payload),
                ]
                return self.send_json(start_job("hive_task_submit", command))
            return self.send_json(start_job("hive_local_task", [
                sys.executable,
                "scripts/hive_node.py",
                "submit",
                "--peer-url",
                "http://127.0.0.1:8791",
                "--kind",
                kind,
                "--payload-json",
                json.dumps(task_payload),
            ]))
        if parsed.path == "/api/capabilities/refresh":
            return self.send_json(start_job("capability_matrix_refresh", [
                sys.executable,
                "scripts/capability_matrix.py",
                "--out",
                "reports/capability_matrix.json",
            ]))
        return self.send_error(HTTPStatus.NOT_FOUND)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def serve_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        for _ in range(3600):
            payload = json.dumps(build_health())
            try:
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            time.sleep(2)

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep dashboard output quiet; status is visible in the UI.
        return


def handle_control(payload: dict[str, Any]) -> dict[str, Any]:
    policy = read_json(ROOT / "configs" / "autonomy_policy.json")
    action = str(payload.get("action") or "")
    profile = str(payload.get("profile") or "inner_loop")
    execute = bool(payload.get("execute"))
    allow_teacher = bool(payload.get("allow_teacher", policy.get("allow_teacher_by_default", True)))
    allow_network_fetch_explicit = "allow_network_fetch" in payload
    allow_network_fetch = bool(
        payload.get("allow_network_fetch", policy.get("allow_network_fetch_by_default", True))
    )
    if execute and profile in LONG_RUN_PROFILES and not payload.get("confirm_long_run"):
        return {
            "ok": False,
            "error": "long_run_confirmation_required",
            "message": f"{profile} is a long-running profile. Confirm it explicitly before launching.",
        }
    if allow_network_fetch_explicit and allow_network_fetch and not payload.get("confirm_external_fetch"):
        return {
            "ok": False,
            "error": "external_fetch_confirmation_required",
            "message": "Network benchmark/data fetches need explicit confirmation.",
        }
    if action == "run_cycle":
        command = [
            sys.executable,
            "scripts/autonomy_cycle.py",
            "--profile",
            profile,
            "--out",
            "reports/autonomy_cycle_last.json",
        ]
        if execute:
            command.append("--execute")
        if allow_teacher:
            command.append("--allow-teacher")
        if allow_network_fetch:
            command.append("--allow-network-fetch")
        return start_job("autonomy_cycle", command)
    if action == "start_daemon":
        command = [
            sys.executable,
            "scripts/sparkstream_daemon.py",
            "--profile",
            profile,
        ]
        if execute:
            command.append("--execute")
        if allow_teacher:
            command.append("--allow-teacher")
        if allow_network_fetch:
            command.append("--allow-network-fetch")
        duration_hours = str(payload.get("duration_hours") or "").strip()
        if duration_hours:
            command.extend(["--duration-hours", duration_hours])
        max_cycles = str(payload.get("max_cycles") or "").strip()
        if max_cycles:
            command.extend(["--max-cycles", max_cycles])
        return start_job("sparkstream_daemon", command)
    if action == "stop_daemon":
        flag = REPORTS / "sparkstream_stop.flag"
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("stop\n", encoding="utf-8")
        return {"ok": True, "status": "stop_flag_written"}
    if action == "pause_daemon":
        flag = REPORTS / "sparkstream_pause.flag"
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("pause\n", encoding="utf-8")
        return {"ok": True, "status": "pause_flag_written"}
    if action == "resume_daemon":
        flag = REPORTS / "sparkstream_pause.flag"
        flag.unlink(missing_ok=True)
        return {"ok": True, "status": "pause_flag_removed"}
    if action == "run_profile":
        command = [
            sys.executable,
            "scripts/run_training_ratchet_profile.py",
            "--profile",
            profile,
            "--out",
            "reports/training_ratchet_profile_run.json",
        ]
        if allow_teacher:
            command.append("--allow-teacher")
        return start_job("training_profile", command)
    if action == "refresh_seeker":
        return start_job("inventory_refresh", [
            sys.executable,
            "scripts/autonomy_cycle.py",
            "--profile",
            "smoke",
            "--out",
            "reports/autonomy_cycle_last.json",
        ])
    return {"ok": False, "error": f"unknown_action:{action}"}


def start_job(name: str, command: list[str]) -> dict[str, Any]:
    existing = existing_singleton_processes(name)
    if existing:
        return {
            "ok": True,
            "status": "already_running",
            "started_new_process": False,
            "name": name,
            "existing_processes": existing[:3],
        }
    job_id = f"job_{int(time.time() * 1000)}"
    log_dir = REPORTS / "sparkstream_jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{job_id}.out.log"
    stderr_path = log_dir / f"{job_id}.err.log"
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(command, cwd=ROOT, text=True, stdout=stdout, stderr=stderr)
    stdout.close()
    stderr.close()
    with ACTIVE_LOCK:
        ACTIVE_JOBS[job_id] = {
            "job_id": job_id,
            "name": name,
            "command": command,
            "pid": proc.pid,
            "started_utc": now(),
            "stdout": str(stdout_path.relative_to(ROOT)).replace("\\", "/"),
            "stderr": str(stderr_path.relative_to(ROOT)).replace("\\", "/"),
            "process": proc,
        }
    return {"ok": True, "job_id": job_id, "name": name, "pid": proc.pid}


def existing_singleton_processes(name: str) -> list[dict[str, Any]]:
    needles = SINGLETON_JOB_NEEDLES.get(name, [])
    if not needles:
        return []
    lowered_needles = [needle.lower().replace("\\", "/") for needle in needles]
    if sys.platform.startswith("win"):
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 2",
        ]
    else:
        command = ["ps", "-eo", "pid=,args="]
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=15)
    except Exception:
        return []
    if result.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    if sys.platform.startswith("win"):
        try:
            payload = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            payload = []
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            pid = int(item.get("ProcessId") or 0)
            command_line = str(item.get("CommandLine") or "")
            if pid and command_line:
                rows.append({"pid": pid, "command_line": command_line})
    else:
        for line in (result.stdout or "").splitlines():
            pid_text, _, command_line = line.strip().partition(" ")
            if pid_text.isdigit() and command_line:
                rows.append({"pid": int(pid_text), "command_line": command_line})
    matches = []
    for row in rows:
        command_line = str(row.get("command_line") or "")
        lowered = command_line.lower().replace("\\", "/")
        if int(row.get("pid") or 0) == 0 or ("get-ciminstance" in lowered and "win32_process" in lowered):
            continue
        if any(needle in lowered for needle in lowered_needles):
            matches.append({"pid": int(row.get("pid") or 0), "command_preview": command_line[:500]})
    matches.sort(key=lambda row: int(row.get("pid") or 0), reverse=True)
    return matches


def run_now(command: list[str], timeout: int = 120) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    return {
        "ok": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "stdout": result.stdout[-8000:],
        "stderr": result.stderr[-8000:],
    }


def build_status() -> dict[str, Any]:
    return {
        "updated_utc": now(),
        "sparkstream": read_json(REPORTS / "sparkstream_status.json"),
        "preflight": summarize_preflight(read_json(REPORTS / "training_preflight_report.json")),
        "candidate": read_json(REPORTS / "candidate_promotion_gate.json"),
        "promotion_closure": read_json(REPORTS / "promotion_closure.json"),
        "accepted_candidate_registry": read_json(REPORTS / "accepted_candidate_registry.json"),
        "frontier_rotation_request": read_json(REPORTS / "frontier_rotation_request.json"),
        "benchmark_lifecycle_overrides": read_json(REPORTS / "benchmark_lifecycle_overrides.json"),
        "benchmarks": read_json(REPORTS / "benchmark_ledger.json"),
        "model_ledger": read_json(REPORTS / "model_ledger.json"),
        "residual_escrow": summarize_residuals(read_json(REPORTS / "residual_escrow.json")),
        "checkpoints": read_json_compact(REPORTS / "checkpoint_registry.json"),
        "checkpoint_backup": read_json(REPORTS / "checkpoint_backup_last.json"),
        "checkpoint_backup_history": read_jsonl_tail(REPORTS / "checkpoint_backup_history.jsonl", 20),
        "benchmark_seeker": read_json(REPORTS / "benchmark_seeker_registry.json"),
        "online_source_catalog": read_json(REPORTS / "online_source_catalog_report.json"),
        "resource_pantry": read_json(REPORTS / "resource_pantry.json"),
        "learning_scoreboard": read_json(REPORTS / "learning_scoreboard.json"),
        "viea_autonomy_spine": read_json(REPORTS / "viea_autonomy_spine.json"),
        "viea_artifact_kernel": read_json(REPORTS / "viea_artifact_kernel.json"),
        "viea_command_executor": read_json(REPORTS / "viea_command_executor.json"),
        "viea_action_executor": read_json(REPORTS / "viea_action_executor.json"),
        "feedback_action_queue": read_json(REPORTS / "feedback_action_queue.json"),
        "broad_transfer_closure": read_json(REPORTS / "broad_transfer_closure.json"),
        "broad_transfer_action_queue": read_json(REPORTS / "broad_transfer_action_queue.json"),
        "repo_repair_main_curriculum": read_json(REPORTS / "repo_repair_main_curriculum.json"),
        "viea_repo_repair_learner": read_json(REPORTS / "viea_repo_repair_learner.json"),
        "symliquid_substrate_map": read_json(REPORTS / "symliquid_substrate_map.json"),
        "symliquid_state_engine_queue": read_json(REPORTS / "symliquid_state_engine_queue.json"),
        "symliquid_state_engine": read_json(REPORTS / "symliquid_state_engine.json"),
        "teacher_architect_loop": read_json(REPORTS / "teacher_architect_loop.json"),
        "teacher_architect_closure": read_json(REPORTS / "teacher_architect_closure.json"),
        "teacher_architect_experiment_runner": read_json(REPORTS / "teacher_architect_experiment_runner.json"),
        "digital_runtime_adapter": read_json(REPORTS / "digital_runtime_adapter.json"),
        "viea_report_map": read_json(REPORTS / "viea_report_map.json"),
        "hive_operator_os": read_json(REPORTS / "hive_operator_os.json"),
        "hive_work_board": read_json(REPORTS / "hive_work_board.json"),
        "hive_work_board_executor": read_json(REPORTS / "hive_work_board_executor.json"),
        "hive_live_command_channel": read_jsonl_tail(REPORTS / "hive_live_command_ledger.jsonl", 20),
        "high_transfer_curriculum_scheduler": read_json(REPORTS / "high_transfer_curriculum_scheduler.json"),
        "hive_node_registry": read_json(REPORTS / "hive_node_registry.json"),
        "hive_morning_report": read_json(REPORTS / "hive_morning_report.json"),
        "hive_overnight_proof": read_json(REPORTS / "hive_overnight_proof.json"),
        "hive_long_run_governor": read_json(REPORTS / "hive_long_run_governor.json"),
        "hive_channel_contract": read_json(REPORTS / "hive_channel_contract.json"),
        "hive_background_tasks": read_json(REPORTS / "hive_background_tasks.json"),
        "hive_persistent_goals": read_json(REPORTS / "hive_persistent_goals.json"),
        "hive_skill_registry": read_json(REPORTS / "hive_skill_registry.json"),
        "hive_tool_hooks": read_json(REPORTS / "hive_tool_hooks.json"),
        "hive_feedback_router": read_json(REPORTS / "hive_feedback_router.json"),
        "hive_execution_safety": read_json(REPORTS / "hive_execution_safety.json"),
        "hive_operator_app_manifest": read_json(REPORTS / "hive_operator_app_manifest.json"),
        "overnight_learning_readiness": read_json(REPORTS / "overnight_learning_readiness.json"),
        "cell_lifecycle": read_json(REPORTS / "cell_lifecycle.json"),
        "cognitive_context_router": read_json(REPORTS / "cognitive_context_router.json"),
        "legacy_port_mechanisms": read_json(REPORTS / "legacy_port_mechanisms.json"),
        "planforge_schedule": read_json(REPORTS / "planforge_schedule.json"),
        "coherence_delirium": read_json(REPORTS / "coherence_delirium_report.json"),
        "proxy_truth_audit": read_json(REPORTS / "proxy_truth_audit.json"),
        "taskspell_contracts": read_json(REPORTS / "taskspell_contracts.json"),
        "low_rank_adapter_bank": read_json(REPORTS / "low_rank_adapter_bank.json"),
        "world_adapter_jobs": read_json(REPORTS / "world_adapter_job_runtime.json"),
        "emulator_game_trace_gateway": read_json(REPORTS / "emulator_game_trace_gateway.json"),
        "compute_mode_acceptance": read_json(REPORTS / "compute_mode_acceptance.json"),
        "orcp_compression_frontier": read_json(REPORTS / "orcp_compression_frontier.json"),
        "device_endpoint_contract": read_json(REPORTS / "device_endpoint_contract.json"),
        "hotpath_quality_gates": read_json(REPORTS / "hotpath_quality_gates.json"),
        "drone_blackbox_parity": read_json(REPORTS / "drone_blackbox_parity.json"),
        "self_mod_proof_bundle": read_json(REPORTS / "self_mod_proof_bundle.json"),
        "first_party_speech_contract": read_json(REPORTS / "first_party_speech_contract.json"),
        "trace_fabric_training_exchange": read_json(REPORTS / "trace_fabric_training_exchange.json"),
        "active_inference_world_model": read_json(REPORTS / "active_inference_world_model.json"),
        "macro_counterexample_gate": read_json(REPORTS / "macro_counterexample_gate.json"),
        "bridge_adapter_native_promotion": read_json(REPORTS / "bridge_adapter_native_promotion.json"),
        "pretraining_readiness_integrity": read_json(REPORTS / "pretraining_readiness_integrity.json"),
        "salience_scheduler": read_json(REPORTS / "salience_scheduler.json"),
        "campaign_dag": read_json(REPORTS / "campaign_dag.json"),
        "dataset_recipe_scaffolder": read_json(REPORTS / "dataset_recipe_scaffolder.json"),
        "evidence_graph_ledger": read_json(REPORTS / "evidence_graph_ledger.json"),
        "runtime_resolution_boundary": read_json(REPORTS / "runtime_resolution_boundary.json"),
        "tiered_memory_consolidation": read_json(REPORTS / "tiered_memory_consolidation.json"),
        "aletheia_advocate_gate": read_json(REPORTS / "aletheia_advocate_gate.json"),
        "synaptic_work_stealing": read_json(REPORTS / "synaptic_work_stealing.json"),
        "architecture_motif_library": read_json(REPORTS / "architecture_motif_library.json"),
        "semantic_intent_repair": read_json(REPORTS / "semantic_intent_repair.json"),
        "eval_track_contract_library": read_json(REPORTS / "eval_track_contract_library.json"),
        "synaptic_permission_decay": read_json(REPORTS / "synaptic_permission_decay.json"),
        "temporal_replay_assertions": read_json(REPORTS / "temporal_replay_assertions.json"),
        "whitecell_threat_memory": read_json(REPORTS / "whitecell_threat_memory.json"),
        "zero_copy_context_prefetch": read_json(REPORTS / "zero_copy_context_prefetch.json"),
        "hil_emulator_gate": read_json(REPORTS / "hil_emulator_gate.json"),
        "formal_runtime_coupling": read_json(REPORTS / "formal_runtime_coupling.json"),
        "veritas_discovery_novelty": read_json(REPORTS / "veritas_discovery_novelty.json"),
        "anti_expert_tribunal_router": read_json(REPORTS / "anti_expert_tribunal_router.json"),
        "probe_router_burst_budget": read_json(REPORTS / "probe_router_burst_budget.json"),
        "rlds_minari_trace_export": read_json(REPORTS / "rlds_minari_trace_export.json"),
        "live_operator_advisors": read_json(REPORTS / "live_operator_advisors.json"),
        "benchmark_bounty_registry": read_json(REPORTS / "benchmark_bounty_registry.json"),
        "legacy_fine_tooth_comb": read_json(REPORTS / "legacy_fine_tooth_comb.json"),
        "knowledge_sources": read_json(REPORTS / "knowledge_source_registry.json"),
        "training_data_sampler": read_json(REPORTS / "training_data_sampler.json"),
        "open_conversation_training_pantry": read_json(REPORTS / "open_conversation_training_pantry.json"),
        "data_inventory": read_json(REPORTS / "training_data_inventory.json"),
        "local_rom_staging": read_json(REPORTS / "local_rom_staging_report.json"),
        "game_asset_inventory": read_json(REPORTS / "game_asset_inventory.json"),
        "local_rom_registry": read_json(REPORTS / "local_rom_registry.json"),
        "synthetic_data": read_json(REPORTS / "synthetic_data_curator.json"),
        "rl_registry": read_json(REPORTS / "rl_benchmark_registry.json"),
        "resource_governor": read_json(REPORTS / "resource_governor.json"),
        "performance_optimizer": read_json(REPORTS / "performance_optimizer.json"),
        "license": license_manager.status_report(write_report=True),
        "compute_market": compute_market.status_report(write_report=True),
        "openai_compat": openai_compat_server.status_report(write_report=True),
        "updates": update_manager.status_report(write_report=True),
        "hive": {
            "status": read_json(REPORTS / "hive_status.json"),
            "peers": read_json(REPORTS / "hive_peers.json"),
            "scheduler": read_json(REPORTS / "hive_scheduler.json"),
            "worker_chunks": read_jsonl_tail(REPORTS / "hive_worker_chunk_ledger.jsonl", 30),
            "relay": read_json(REPORTS / "hive_relay_status.json"),
            "public_contribution": read_json(REPORTS / "public_hive_contribution_status.json"),
            "tasks": read_jsonl_tail(REPORTS / "hive_task_ledger.jsonl", 30),
            "queue": read_jsonl_tail(REPORTS / "hive_task_queue.jsonl", 30),
        },
        "autonomous_goal": read_json(REPORTS / "autonomous_goal_last.json"),
        "arm_lifecycle_governance": read_json(REPORTS / "arm_lifecycle_governance.json"),
        "arm_sucker_registry": read_json(REPORTS / "arm_sucker_registry.json"),
        "launch_readiness": read_json(REPORTS / "autonomy_launch_readiness.json"),
        "history": read_json_compact(REPORTS / "sparkstream_history.json"),
        "context_packets": read_json_compact(REPORTS / "context_packet_ledger.json"),
        "virtual_context_memory": read_json(REPORTS / "virtual_context_memory_probe.json"),
        "virtual_context_memory_status": read_json(REPORTS / "virtual_context_memory_status.json"),
        "virtual_context_memory_bench": read_json(REPORTS / "virtual_context_memory_bench.json"),
        "virtual_context_memory_consumer_audit": read_json(REPORTS / "virtual_context_memory_consumer_audit.json"),
        "virtual_context_memory_graph": read_json_compact(REPORTS / "virtual_context_memory_graph.json"),
        "virtual_context_compiled_context": read_json_compact(REPORTS / "virtual_context_compiled_context.json"),
        "capability_matrix": read_json(REPORTS / "capability_matrix.json"),
        "benchmaxx_curriculum": read_json(REPORTS / "benchmaxx_curriculum.json"),
        "benchmark_adapter_factory": read_json(REPORTS / "benchmark_adapter_factory.json"),
        "benchmark_pantry_unblocker": read_json(REPORTS / "benchmark_pantry_unblocker.json"),
        "candidate_bottleneck_reducer": read_json(REPORTS / "candidate_bottleneck_reducer.json"),
        "ai_grand_prix_spec": read_json(REPORTS / "ai_grand_prix_spec_digest.json"),
        "minecraft_runtime_probe": read_json(REPORTS / "minecraft_runtime_probe.json"),
        "python_runtime_compatibility": read_json(REPORTS / "python_runtime_compatibility.json"),
        "architecture_experiments": read_json(REPORTS / "architecture_experiment_governance.json"),
        "architecture_experiment_runner": read_json(REPORTS / "architecture_experiment_runner.json"),
        "autoresearch_gap_audit": read_json(REPORTS / "autoresearch_gap_audit.json"),
        "autoresearch_experiment_summary": read_json(REPORTS / "autoresearch_experiment_ledger_summary.json"),
        "autoresearch_experiment_ledger": read_jsonl_tail(REPORTS / "autoresearch_experiment_ledger.jsonl", 50),
        "loop_closure_harvester": read_json(REPORTS / "loop_closure_harvester.json"),
        "native_voice_training_manifest": read_json(REPORTS / "native_voice_training_manifest.json"),
        "native_stt_decoder": read_json(REPORTS / "native_stt_decoder.json"),
        "native_tts_generator": read_json(REPORTS / "native_tts_generator.json"),
        "native_voice_io": read_json(REPORTS / "native_voice_io.json"),
        "transfer_eval_suite": read_json(REPORTS / "transfer_eval_suite.json"),
        "arm_transfer_plan": read_json(REPORTS / "arm_transfer_plan.json"),
        "arm_transfer_artifacts": read_json(REPORTS / "arm_transfer_artifacts.json"),
        "self_evolution_governance": read_json(REPORTS / "self_evolution_governance.json"),
        "teacher_self_edit": read_json(REPORTS / "teacher_self_edit_last.json"),
        "teacher_self_edit_proof": read_json(REPORTS / "teacher_self_edit_proof.json"),
        "attd": read_json(REPORTS / "attd_report.json"),
        "attd_maintenance_packets": read_json(REPORTS / "attd_maintenance_packets.json"),
        "external_inference_audit": read_json(REPORTS / "external_inference_audit.json"),
        "frontier_policy": read_json(REPORTS / "frontier_policy_status.json"),
        "autonomy_watchdog": read_json(REPORTS / "autonomy_watchdog.json"),
        "personality_core": read_json(REPORTS / "personality_core.json"),
        "personality_context": read_json(REPORTS / "personality_context_last.json"),
        "personality_drift_eval": read_json(REPORTS / "personality_drift_eval.json"),
        "personality_runtime_audit": read_json(REPORTS / "personality_runtime_audit.json"),
        "belief_update_governance": read_json(REPORTS / "belief_update_governance.json"),
        "checkpoint_chat": read_json(REPORTS / "checkpoint_chat_last.json"),
        "teacher": {
            "last": read_json(REPORTS / "teacher_oracle_last.json"),
            "calls": read_jsonl_tail(REPORTS / "teacher_calls.jsonl", 20),
            "queue": read_jsonl_tail(REPORTS / "teacher_request_queue.jsonl", 20),
            "self_edit_traces": read_jsonl_tail(REPORTS / "teacher_self_edit_traces.jsonl", 20),
        },
        "autonomy": {
            "last": read_json(REPORTS / "autonomy_cycle_last.json"),
            "ledger_tail": read_jsonl_tail(REPORTS / "autonomy_ledger.jsonl", 50),
            "queue": read_json(REPORTS / "self_improvement_queue.json"),
            "daemon_ledger_tail": read_jsonl_tail(REPORTS / "sparkstream_daemon_ledger.jsonl", 50),
            "goal_ledger_tail": read_jsonl_tail(REPORTS / "autonomous_goal_ledger.jsonl", 30),
            "real_routing_trace_tail": read_jsonl_tail(REPORTS / "routing_memory_real_traces.jsonl", 30),
            "pause_flag": (REPORTS / "sparkstream_pause.flag").exists(),
            "stop_flag": (REPORTS / "sparkstream_stop.flag").exists(),
        },
        "jobs": active_jobs(),
        "reports": recent_reports(),
    }


def build_health() -> dict[str, Any]:
    """Small liveness payload for LAN probes and server-sent events.

    `/api/status` is a full operator cockpit and can legitimately include many
    reports. Health must stay tiny so supervisors do not mistake a large status
    payload for a dead dashboard.
    """
    return {
        "ok": True,
        "policy": "project_theseus_dashboard_health_v1",
        "updated_utc": now(),
        "sparkstream": status_ref(REPORTS / "sparkstream_status.json"),
        "hive_work_board_executor": status_ref(REPORTS / "hive_work_board_executor.json"),
        "hive_long_run_governor": status_ref(REPORTS / "hive_long_run_governor.json"),
        "hive_overnight_proof": status_ref(REPORTS / "hive_overnight_proof.json"),
        "broad_transfer_matrix": status_ref(REPORTS / "broad_transfer_matrix.json"),
        "jobs": active_jobs(),
    }


def active_jobs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ACTIVE_LOCK:
        for job_id, job in list(ACTIVE_JOBS.items()):
            proc = job["process"]
            returncode = proc.poll()
            row = {key: value for key, value in job.items() if key != "process"}
            row["returncode"] = returncode
            row["running"] = returncode is None
            rows.append(row)
            if returncode is not None:
                # Keep completed jobs visible for the current dashboard process.
                job["returncode"] = returncode
    return sorted(rows, key=lambda item: item.get("started_utc", ""), reverse=True)[:30]


def summarize_preflight(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    return {
        "heavy_training_allowed": report.get("heavy_training_allowed"),
        "passed": report.get("passed"),
        "total": report.get("total"),
        "blocker_count": report.get("blocker_count"),
        "warning_count": report.get("warning_count"),
        "warnings": report.get("warnings", []),
        "blockers": report.get("blockers", []),
    }


def summarize_residuals(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    return {
        "summary": report.get("summary", {}),
        "clusters": (report.get("clusters") or [])[:30],
    }


def recent_reports() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not REPORTS.exists():
        return rows
    for path in REPORTS.glob("*.json"):
        stat = path.stat()
        rows.append(
            {
                "name": path.name,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": stat.st_size,
                "modified": stat.st_mtime,
            }
        )
    return sorted(rows, key=lambda item: item["modified"], reverse=True)[:80]


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_json_compact(path: Path, max_bytes: int = 1_000_000) -> Any:
    if not path.exists():
        return {}
    try:
        stat = path.stat()
    except OSError:
        return {}
    if stat.st_size > max_bytes:
        return {
            "policy": "compact_dashboard_report_reference_v1",
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": stat.st_size,
            "modified": stat.st_mtime,
            "truncated_for_dashboard": True,
            "reason": "large_report_available_on_disk",
        }
    return read_json(path)


def status_ref(path: Path) -> dict[str, Any]:
    payload = read_json_compact(path, max_bytes=256_000)
    if not isinstance(payload, dict):
        return {}
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "policy": payload.get("policy"),
        "trigger_state": payload.get("trigger_state"),
        "created_utc": payload.get("created_utc") or payload.get("updated_utc"),
        "summary": payload.get("summary"),
        "truncated_for_dashboard": payload.get("truncated_for_dashboard", False),
    }


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
