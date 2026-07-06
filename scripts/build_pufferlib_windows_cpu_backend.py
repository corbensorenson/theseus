"""Build a Windows-native PufferLib CPU backend for one Ocean environment.

PufferLib's upstream build.sh is Linux/mac oriented. For Theseus on Windows we
only need the CPU `_C.pyd` admission path first: native reset/step buffers,
PufferLib policy training, and governed trace evidence. This builder keeps the
output on D: and installs only the final extension into the vendored editable
PufferLib package.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "pufferlib"
SRC = VENDOR / "src"
PUFFER_PKG = VENDOR / "pufferlib"
D_BUILD = Path("D:/ProjectTheseus/runtime/pufferlib4/windows_cpu_build")
DEFAULT_REPORT = ROOT / "reports" / "pufferlib4_windows_cpu_build.json"
DEFAULT_ENV = "cartpole"
PORTABLE_CARTPOLE_SOURCE = ROOT / "adapters" / "pufferlib" / "pufferlib_windows_cpu_cartpole_backend.cpp"


RAYLIB_STUB = r"""
#pragma once
#include <stdarg.h>
#include <stdio.h>

#ifndef PI
#define PI 3.14159265358979323846f
#endif

typedef struct Color { unsigned char r, g, b, a; } Color;
typedef struct Vector2 { float x, y; } Vector2;
typedef struct Rectangle { float x, y, width, height; } Rectangle;
typedef struct Texture2D { unsigned int id; int width; int height; int mipmaps; int format; } Texture2D;

static const Color WHITE = {255, 255, 255, 255};
static const Color DARKGRAY = {80, 80, 80, 255};

#define KEY_RIGHT 262
#define KEY_D 68
#define KEY_LEFT_SHIFT 340
#define KEY_ESCAPE 256
#define KEY_TAB 258

static inline int IsKeyDown(int key) { (void)key; return 0; }
static inline int IsKeyPressed(int key) { (void)key; return 0; }
static inline int IsWindowReady(void) { return 0; }
static inline int WindowShouldClose(void) { return 1; }
static inline void InitWindow(int width, int height, const char* title) { (void)width; (void)height; (void)title; }
static inline void CloseWindow(void) {}
static inline void SetTargetFPS(int fps) { (void)fps; }
static inline void ToggleFullscreen(void) {}
static inline void BeginDrawing(void) {}
static inline void EndDrawing(void) {}
static inline void ClearBackground(Color color) { (void)color; }
static inline void DrawLine(int startPosX, int startPosY, int endPosX, int endPosY, Color color) { (void)startPosX; (void)startPosY; (void)endPosX; (void)endPosY; (void)color; }
static inline void DrawLineEx(Vector2 startPos, Vector2 endPos, float thick, Color color) { (void)startPos; (void)endPos; (void)thick; (void)color; }
static inline void DrawRectangle(int posX, int posY, int width, int height, Color color) { (void)posX; (void)posY; (void)width; (void)height; (void)color; }
static inline void DrawText(const char* text, int posX, int posY, int fontSize, Color color) { (void)text; (void)posX; (void)posY; (void)fontSize; (void)color; }
static inline Texture2D LoadTexture(const char* fileName) { (void)fileName; Texture2D t = {0, 0, 0, 0, 0}; return t; }
static inline void UnloadTexture(Texture2D texture) { (void)texture; }
static inline void DrawTexturePro(Texture2D texture, Rectangle source, Rectangle dest, Vector2 origin, float rotation, Color tint) { (void)texture; (void)source; (void)dest; (void)origin; (void)rotation; (void)tint; }
static inline const char* TextFormat(const char* fmt, ...) {
    static char buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    return buf;
}
"""


STDATOMIC_STUB = r"""
#pragma once
typedef int atomic_int;
static inline int atomic_load(const atomic_int* obj) { return *obj; }
static inline void atomic_store(atomic_int* obj, int desired) { *obj = desired; }
"""


PTHREAD_STUB = r"""
#pragma once
#include <windows.h>

typedef HANDLE pthread_t;

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
struct timespec { long tv_sec; long tv_nsec; };
static inline int clock_gettime(int clock_id, struct timespec* ts) {
    (void)clock_id;
    static LARGE_INTEGER freq;
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&counter);
    double seconds = (double)counter.QuadPart / (double)freq.QuadPart;
    ts->tv_sec = (long)seconds;
    ts->tv_nsec = (long)((seconds - (double)ts->tv_sec) * 1000000000.0);
    return 0;
}
#endif

static inline int pthread_create(pthread_t* thread, const void* attr, void* (*start_routine)(void*), void* arg) {
    (void)attr;
    *thread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)start_routine, arg, 0, NULL);
    return *thread == NULL ? -1 : 0;
}

static inline int pthread_join(pthread_t thread, void** retval) {
    (void)retval;
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return 0;
}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--out", default=str(DEFAULT_REPORT.relative_to(ROOT)))
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--upstream-vecenv-build", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report_path = resolve(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    env_name = args.env.strip()
    build_dir = D_BUILD / env_name
    include_dir = build_dir / "include"
    obj_dir = build_dir / "obj"
    include_dir.mkdir(parents=True, exist_ok=True)
    obj_dir.mkdir(parents=True, exist_ok=True)
    write_compat_headers(include_dir)

    py = Path(sys.executable)
    py_include = Path(sysconfig.get_path("include"))
    py_lib = Path(sysconfig.get_config_var("LIBDIR") or py_include.parent / "libs")
    ext_suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or ".pyd")
    target = PUFFER_PKG / f"_C{ext_suffix}"
    pybind_include = query_python(py, "import pybind11; print(pybind11.get_include())")
    numpy_include = query_python(py, "import numpy; print(numpy.get_include())")
    if not pybind_include or not numpy_include:
        return finish(
            report_path,
            started,
            "RED",
            [{"id": "missing_python_build_includes", "detail": "pybind11 or numpy include path could not be resolved"}],
            {},
        )

    env = msvc_environment()
    if not env:
        return finish(report_path, started, "RED", [{"id": "msvc_environment_missing", "detail": "VsDevCmd/vcvars64 could not be loaded"}], {})

    details: dict[str, Any] = {"env_name": env_name, "target": rel(target)}
    if env_name == "cartpole" and not args.upstream_vecenv_build:
        if not PORTABLE_CARTPOLE_SOURCE.exists():
            return finish(
                report_path,
                started,
                "RED",
                [{"id": "portable_cartpole_source_missing", "detail": str(PORTABLE_CARTPOLE_SOURCE)}],
                details,
            )
        cpu_obj = obj_dir / "pufferlib_windows_cpu_cartpole_backend.obj"
        commands = [
            [
                "cl",
                "/nologo",
                "/O2",
                "/MD",
                "/EHsc",
                "/std:c++17",
                f"/I{py_include}",
                f"/I{pybind_include}",
                f"/I{numpy_include}",
                "/D_CRT_SECURE_NO_WARNINGS",
                f"/Fo{cpu_obj}",
                "/c",
                str(PORTABLE_CARTPOLE_SOURCE),
            ],
            [
                "link",
                "/nologo",
                "/DLL",
                "/INCREMENTAL:NO",
                f"/OUT:{target}",
                str(cpu_obj),
                f"/LIBPATH:{py_lib}",
                f"python{sys.version_info.major}{sys.version_info.minor}.lib",
            ],
        ]
        details.update(
            {
                "build_mode": "windows_portable_cartpole_native_backend",
                "source": rel(PORTABLE_CARTPOLE_SOURCE),
                "obs_tensor": "FloatTensor",
            }
        )
    else:
        binding = VENDOR / "ocean" / env_name / "binding.c"
        if not binding.exists():
            return finish(report_path, started, "RED", [{"id": "missing_binding_c", "detail": str(binding)}], details)
        binding_text = binding.read_text(encoding="utf-8", errors="replace")
        if 'vecenv.h' not in binding_text:
            return finish(
                report_path,
                started,
                "RED",
                [{"id": "unsupported_legacy_env_binding", "detail": f"{env_name} uses legacy env_binding.h, not vecenv.h"}],
                {"binding": rel(binding), **details},
            )
        obs_tensor = parse_define(binding_text, "OBS_TENSOR_T") or "FloatTensor"
        binding_obj = obj_dir / "binding.obj"
        cpu_obj = obj_dir / "bindings_cpu.obj"
        commands = [
            [
                "cl",
                "/nologo",
                "/O2",
                "/MD",
                "/TC",
                "/std:c11",
                f"/I{include_dir}",
                f"/I{VENDOR}",
                f"/I{SRC}",
                f"/I{binding.parent}",
                "/DPLATFORM_DESKTOP",
                "/D_CRT_SECURE_NO_WARNINGS",
                "/D_USE_MATH_DEFINES",
                f"/Fo{binding_obj}",
                "/c",
                str(binding),
            ],
            [
                "cl",
                "/nologo",
                "/O2",
                "/MD",
                "/EHsc",
                "/std:c++17",
                f"/I{include_dir}",
                f"/I{VENDOR}",
                f"/I{SRC}",
                f"/I{binding.parent}",
                f"/I{py_include}",
                f"/I{pybind_include}",
                f"/I{numpy_include}",
                "/DPLATFORM_DESKTOP",
                "/DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
                f"/DOBS_TENSOR_T={obs_tensor}",
                f"/DENV_NAME={env_name}",
                "/DPRECISION_FLOAT",
                f"/Fo{cpu_obj}",
                "/c",
                str(SRC / "bindings_cpu.cpp"),
            ],
            [
                "link",
                "/nologo",
                "/DLL",
                "/INCREMENTAL:NO",
                f"/OUT:{target}",
                str(cpu_obj),
                str(binding_obj),
                f"/LIBPATH:{py_lib}",
                f"python{sys.version_info.major}{sys.version_info.minor}.lib",
            ],
        ]
        details.update(
            {
                "build_mode": "upstream_vecenv_cpu_backend",
                "binding": rel(binding),
                "obs_tensor": obs_tensor,
            }
        )

    results = []
    blockers: list[dict[str, Any]] = []
    for command in commands:
        resolved_tool = shutil.which(command[0], path=env.get("PATH", ""))
        if resolved_tool:
            command = [resolved_tool] + command[1:]
        result = run(command, env=env, cwd=ROOT, timeout=120)
        results.append(compact_result(result))
        if result.returncode != 0:
            blockers.append(
                {
                    "id": "windows_cpu_backend_compile_failed",
                    "detail": f"{command[0]} returned {result.returncode}: {tail(result.stderr + result.stdout, 1600)}",
                }
            )
            if not args.keep_going:
                break

    import_probe: dict[str, Any] = {}
    if not blockers:
        import_probe = probe_import(py, env_name)
        if not import_probe.get("ok"):
            blockers.append({"id": "windows_cpu_backend_import_failed", "detail": import_probe.get("error") or import_probe.get("stderr_tail")})

    state = "GREEN" if not blockers and target.exists() else "RED"
    return finish(
        report_path,
        started,
        state,
        blockers,
        {
            **details,
            "build_dir": str(build_dir),
            "target_exists": target.exists(),
            "commands": results,
            "import_probe": import_probe,
        },
    )


def write_compat_headers(include_dir: Path) -> None:
    (include_dir / "raylib.h").write_text(RAYLIB_STUB.strip() + "\n", encoding="utf-8")
    (include_dir / "stdatomic.h").write_text(STDATOMIC_STUB.strip() + "\n", encoding="utf-8")
    (include_dir / "pthread.h").write_text(PTHREAD_STUB.strip() + "\n", encoding="utf-8")


def parse_define(text: str, name: str) -> str:
    match = re.search(rf"^\s*#\s*define\s+{re.escape(name)}\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE)
    return match.group(1) if match else ""


def msvc_environment() -> dict[str, str]:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    install = ""
    if vswhere.exists():
        proc = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
            capture_output=True,
            timeout=30,
        )
        install = proc.stdout.strip()
    candidates = [
        Path(install) / "Common7" / "Tools" / "VsDevCmd.bat" if install else None,
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"),
    ]
    devcmd = next((path for path in candidates if path and path.exists()), None)
    if not devcmd:
        return {}
    D_BUILD.mkdir(parents=True, exist_ok=True)
    loader = D_BUILD / "load_msvc_env.cmd"
    loader.write_text(f'@echo off\r\ncall "{devcmd}" -arch=x64 -host_arch=x64 >nul\r\nset\r\n', encoding="utf-8")
    proc = subprocess.run(
        ["cmd.exe", "/d", "/c", str(loader)],
        text=True,
        capture_output=True,
        timeout=45,
    )
    if proc.returncode != 0:
        return {}
    env = os.environ.copy()
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key] = value
    if env.get("Path"):
        env["PATH"] = env["Path"]
    return env


def query_python(py: Path, code: str) -> str:
    result = subprocess.run([str(py), "-c", code], cwd=ROOT, text=True, capture_output=True, timeout=30)
    return result.stdout.strip() if result.returncode == 0 else ""


def probe_import(py: Path, env_name: str) -> dict[str, Any]:
    code = f"""
import importlib, json, sys
try:
    mod = importlib.import_module('pufferlib._C')
    payload = {{
        'ok': True,
        'path': getattr(mod, '__file__', ''),
        'env_name': getattr(mod, 'env_name', ''),
        'gpu': getattr(mod, 'gpu', None),
        'precision_bytes': getattr(mod, 'precision_bytes', None),
        'has_create_vec': hasattr(mod, 'create_vec'),
        'expected_env': {env_name!r},
    }}
except Exception as exc:
    payload = {{'ok': False, 'error': f'{{type(exc).__name__}}: {{exc}}'}}
print(json.dumps(payload, sort_keys=True))
"""
    result = run([str(py), "-c", code], env=os.environ.copy(), cwd=ROOT, timeout=30)
    payload = parse_json_output(result.stdout)
    if not isinstance(payload, dict):
        payload = {"ok": False, "stdout_tail": tail(result.stdout, 1000), "stderr_tail": tail(result.stderr, 1000)}
    return payload


def run(command: list[str], *, env: dict[str, str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)


def compact_result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "command": result.args,
        "returncode": result.returncode,
        "stdout_tail": tail(result.stdout, 1600),
        "stderr_tail": tail(result.stderr, 1600),
    }


def finish(report_path: Path, started: float, state: str, blockers: list[dict[str, Any]], details: dict[str, Any]) -> int:
    payload = {
        "policy": "project_theseus_pufferlib4_windows_cpu_backend_build_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trigger_state": state,
        "summary": {
            "windows_native_backend_built": state == "GREEN",
            "blocker_count": len(blockers),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "blockers": blockers,
        "details": details,
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if state == "GREEN" else 2


def parse_json_output(text: str) -> Any:
    for line in reversed((text or "").splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def tail(text: str, chars: int) -> str:
    return (text or "")[-chars:]


if __name__ == "__main__":
    raise SystemExit(main())
