use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use clap::{Parser, Subcommand};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(name = "theseus-supervisor")]
#[command(about = "Native Project Theseus launcher, doctor, and installer supervisor.")]
struct Cli {
    #[command(subcommand)]
    command: SupervisorCommand,
}

#[derive(Debug, Subcommand)]
enum SupervisorCommand {
    Doctor {
        #[arg(long)]
        json: bool,
    },
    InitRuntime {
        #[arg(long, default_value = "")]
        runtime_root: String,
    },
    Setup {
        #[arg(long, default_value_t = 8788)]
        port: u16,
        #[arg(long)]
        no_open: bool,
    },
    Start {
        #[arg(long)]
        relay: bool,
        #[arg(long)]
        restart: bool,
    },
    BuildCli {
        #[arg(long)]
        release: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    let root =
        find_root().unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let result = match cli.command {
        SupervisorCommand::Doctor { json } => doctor(&root, json),
        SupervisorCommand::InitRuntime { runtime_root } => init_runtime(&root, &runtime_root),
        SupervisorCommand::Setup { port, no_open } => setup(&root, port, no_open),
        SupervisorCommand::Start { relay, restart } => start(&root, relay, restart),
        SupervisorCommand::BuildCli { release } => build_cli(&root, release),
    };
    match result {
        Ok(value) => {
            println!("{}", serde_json::to_string_pretty(&value).unwrap());
        }
        Err(error) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "ok": false,
                    "error": error,
                }))
                .unwrap()
            );
            std::process::exit(2);
        }
    }
}

fn doctor(root: &Path, json_output: bool) -> Result<serde_json::Value, String> {
    let python = python_command(root);
    let runtime = run_capture(
        root,
        &python,
        &["scripts/runtime_paths.py", "status", "--create"],
        None,
    )?;
    let checks = json!({
        "python": command_version(root, &python, &["--version"]),
        "cargo": command_version(root, "cargo", &["--version"]),
        "nvidia_smi": command_version(root, "nvidia-smi", &["--version"]),
        "nvcc": command_version(root, "nvcc", &["--version"]),
        "mlx": python_probe(root, &python, "import mlx.core as mx; print('mlx.core ok')"),
    });
    let ok = checks["python"]["ok"].as_bool().unwrap_or(false)
        && checks["cargo"]["ok"].as_bool().unwrap_or(false);
    let value = json!({
        "ok": ok,
        "policy": "project_theseus_supervisor_doctor_v0",
        "root": root,
        "runtime_paths": parse_json_or_text(&runtime),
        "checks": checks,
        "installer_boundaries": {
            "native_supervisor": true,
            "signed_windows_installer": false,
            "notarized_macos_pkg": false,
            "linux_appimage_deb_rpm": false,
            "note": "This binary is the native supervisor entrypoint. Signing/notarization is performed by release packaging outside the source checkout."
        }
    });
    if !json_output {
        eprintln!("Project Theseus supervisor doctor complete.");
    }
    Ok(value)
}

fn init_runtime(root: &Path, runtime_root: &str) -> Result<serde_json::Value, String> {
    let python = python_command(root);
    let mut args = vec!["scripts/runtime_paths.py", "init"];
    let mut owned = Vec::new();
    if !runtime_root.is_empty() {
        args.push("--runtime-root");
        owned.push(runtime_root.to_string());
    }
    let mut final_args = args.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
    final_args.extend(owned);
    let out = run_capture_owned(root, &python, &final_args, None)?;
    Ok(parse_json_or_text(&out))
}

fn setup(root: &Path, port: u16, no_open: bool) -> Result<serde_json::Value, String> {
    let python = python_command(root);
    let mut args = vec![
        "scripts/theseus_setup_wizard.py".to_string(),
        "--port".to_string(),
        port.to_string(),
    ];
    if !no_open {
        args.push("--open".to_string());
    }
    spawn(root, &python, &args)?;
    Ok(json!({
        "ok": true,
        "policy": "project_theseus_supervisor_setup_v0",
        "url": format!("http://127.0.0.1:{port}"),
    }))
}

fn start(root: &Path, relay: bool, restart: bool) -> Result<serde_json::Value, String> {
    if cfg!(windows) {
        let mut args = vec![
            "-ExecutionPolicy".to_string(),
            "Bypass".to_string(),
            "-File".to_string(),
            "scripts\\start_theseus_hive.ps1".to_string(),
        ];
        if relay {
            args.push("-StartRelay".to_string());
        }
        if restart {
            args.push("-Restart".to_string());
        }
        spawn(root, "powershell", &args)?;
    } else {
        let mut args = Vec::new();
        if relay {
            args.push("THESEUS_START_RELAY=1".to_string());
        }
        if restart {
            args.push("THESEUS_RESTART=1".to_string());
        }
        args.push("./scripts/start_theseus_hive.sh".to_string());
        spawn(root, "sh", &["-c".to_string(), args.join(" ")])?;
    }
    Ok(json!({
        "ok": true,
        "policy": "project_theseus_supervisor_start_v0",
        "dashboard_url": "http://127.0.0.1:8787",
        "hive_status_url": "http://127.0.0.1:8791/api/hive/status",
    }))
}

fn build_cli(root: &Path, release: bool) -> Result<serde_json::Value, String> {
    let mut args = vec![
        "build".to_string(),
        "-p".to_string(),
        "symliquid-cli".to_string(),
    ];
    if release {
        args.push("--release".to_string());
    }
    let out = run_capture_owned(root, "cargo", &args, None)?;
    Ok(json!({
        "ok": true,
        "policy": "project_theseus_supervisor_build_cli_v0",
        "release": release,
        "stdout_tail": tail(&out, 2000),
    }))
}

fn find_root() -> Option<PathBuf> {
    let mut dir = env::current_dir().ok()?;
    loop {
        if dir.join("Cargo.toml").exists() && dir.join("scripts").exists() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn python_command(root: &Path) -> String {
    let candidate = if cfg!(windows) {
        root.join(".venv-puffer").join("Scripts").join("python.exe")
    } else {
        root.join(".venv-puffer").join("bin").join("python")
    };
    if candidate.exists() {
        candidate.to_string_lossy().to_string()
    } else if cfg!(windows) {
        "py".to_string()
    } else {
        "python3".to_string()
    }
}

fn command_version(root: &Path, command: &str, args: &[&str]) -> serde_json::Value {
    match run_capture(root, command, args, Some(10)) {
        Ok(out) => json!({"ok": true, "stdout": tail(&out, 600)}),
        Err(error) => json!({"ok": false, "error": error}),
    }
}

fn python_probe(root: &Path, python: &str, code: &str) -> serde_json::Value {
    match run_capture(root, python, &["-c", code], Some(10)) {
        Ok(out) => json!({"ok": true, "stdout": tail(&out, 600)}),
        Err(error) => json!({"ok": false, "error": error}),
    }
}

fn run_capture(
    root: &Path,
    command: &str,
    args: &[&str],
    _timeout_seconds: Option<u64>,
) -> Result<String, String> {
    let output = Command::new(command)
        .args(args)
        .current_dir(root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| format!("{command}: {error}"))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

fn run_capture_owned(
    root: &Path,
    command: &str,
    args: &[String],
    timeout_seconds: Option<u64>,
) -> Result<String, String> {
    let borrowed = args.iter().map(String::as_str).collect::<Vec<_>>();
    run_capture(root, command, &borrowed, timeout_seconds)
}

fn spawn(root: &Path, command: &str, args: &[String]) -> Result<(), String> {
    Command::new(command)
        .args(args)
        .current_dir(root)
        .spawn()
        .map_err(|error| format!("{command}: {error}"))?;
    Ok(())
}

fn parse_json_or_text(raw: &str) -> serde_json::Value {
    serde_json::from_str(raw).unwrap_or_else(|_| json!({"ok": true, "raw": tail(raw, 4000)}))
}

fn tail(value: &str, max_chars: usize) -> String {
    let chars = value.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        value.to_string()
    } else {
        chars[chars.len() - max_chars..].iter().collect()
    }
}
