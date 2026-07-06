param(
  [string]$Config = "configs/windows_sandbox_runtime.json",
  [string]$Out = "reports/podman_sandbox_setup.json",
  [string]$MarkdownOut = "reports/podman_sandbox_setup.md",
  [switch]$RunContainerSmoke
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Write-JsonFile {
  param($Path, $Payload)
  $full = Join-Path $Root $Path
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $full) | Out-Null
  $json = $Payload | ConvertTo-Json -Depth 12
  [System.IO.File]::WriteAllText($full, $json + "`n", [System.Text.UTF8Encoding]::new($false))
}

function Join-CommandArguments {
  param([string[]]$Arguments)
  $parts = @()
  foreach ($arg in $Arguments) {
    $text = [string]$arg
    if ($text -match '[\s"]') {
      $text = '"' + ($text -replace '"', '\"') + '"'
    }
    $parts += $text
  }
  return ($parts -join " ")
}

function Run-Step {
  param(
    [string]$Name,
    [string]$FilePath,
    [string[]]$Arguments,
    [int]$TimeoutSeconds = 120
  )
  $started = Get-Date
  $command = @($FilePath) + $Arguments
  try {
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FilePath
    $psi.WorkingDirectory = $Root
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.Arguments = Join-CommandArguments -Arguments $Arguments
    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    [void]$proc.Start()
    if (-not $proc.WaitForExit($TimeoutSeconds * 1000)) {
      try {
        $proc.Kill()
      } catch {
      }
      return [ordered]@{
        name = $Name
        ok = $false
        timed_out = $true
        exit_code = -2
        command = $command
        runtime_seconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 3)
        error = "timeout_after_${TimeoutSeconds}s"
      }
    }
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    return [ordered]@{
      name = $Name
      ok = ($proc.ExitCode -eq 0)
      exit_code = $proc.ExitCode
      command = $command
      runtime_seconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 3)
      stdout_tail = if ($stdout.Length -gt 2000) { $stdout.Substring($stdout.Length - 2000) } else { $stdout }
      stderr_tail = if ($stderr.Length -gt 2000) { $stderr.Substring($stderr.Length - 2000) } else { $stderr }
    }
  } catch {
    return [ordered]@{
      name = $Name
      ok = $false
      exit_code = -1
      command = $command
      runtime_seconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 3)
      error = $_.Exception.Message
    }
  }
}

$cfgPath = Join-Path $Root $Config
$cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
$podmanDir = Split-Path -Parent $cfg.podman_bin
$env:Path = "$podmanDir;$env:Path"

$dirs = @(
  $cfg.storage_root,
  $cfg.container_tmp,
  "D:/ProjectTheseus/podman_machine"
)
foreach ($dir in $dirs) {
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$steps = @()
$checks = [ordered]@{
  podman_bin = $cfg.podman_bin
  podman_bin_exists = [bool](Test-Path $cfg.podman_bin)
  wsl_command_exists = [bool](Get-Command wsl.exe -ErrorAction SilentlyContinue)
  config = $Config
}

if (-not $checks.podman_bin_exists) {
  $payload = [ordered]@{
    policy = "project_theseus_podman_sandbox_setup_v0"
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
    ok = $false
    status = "podman_bin_missing"
    checks = $checks
    steps = $steps
    next_actions = @("Install Podman to D:\ProjectTheseus\tools\podman with winget, then rerun this script.")
  }
  Write-JsonFile $Out $payload
  throw "Podman binary missing: $($cfg.podman_bin)"
}

$steps += Run-Step -Name "podman_version" -FilePath $cfg.podman_bin -Arguments @("--version")
$wslPath = "C:\Windows\System32\wsl.exe"
if ($checks.wsl_command_exists) {
  $wslList = Run-Step -Name "wsl_list" -FilePath $wslPath -Arguments @("--list", "--quiet") -TimeoutSeconds 45
  $steps += $wslList
} else {
  $wslList = [ordered]@{ name = "wsl_list"; ok = $false; exit_code = -1; error = "wsl.exe_missing" }
}

if (-not $wslList.ok) {
  $payload = [ordered]@{
    policy = "project_theseus_podman_sandbox_setup_v0"
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
    ok = $false
    status = "blocked_reboot_required_after_wsl_install"
    checks = $checks
    steps = $steps
    safety = [ordered]@{
      network_during_scoring = "forbidden"
      public_benchmark_training_use_allowed = $false
      external_inference_calls = 0
      teacher_apply_mode_allowed = $false
    }
    next_actions = @("Reboot Windows so WSL feature installation can complete, then rerun scripts/setup_podman_sandbox_windows.ps1.")
  }
  Write-JsonFile $Out $payload
  $mdPath = Join-Path $Root $MarkdownOut
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $mdPath) | Out-Null
  $md = @(
    "# Podman Sandbox Setup",
    "",
    ("Created: " + $payload.created_utc),
    "",
    "- status: blocked_reboot_required_after_wsl_install",
    "- podman: " + $cfg.podman_bin,
    "",
    "## Next Actions",
    "",
    "- Reboot Windows so WSL feature installation can complete, then rerun scripts/setup_podman_sandbox_windows.ps1."
  ) -join "`n"
  [System.IO.File]::WriteAllText($mdPath, $md + "`n", [System.Text.UTF8Encoding]::new($false))
  $payload | ConvertTo-Json -Depth 12
  exit 0
}
$machineList = Run-Step -Name "podman_machine_list" -FilePath $cfg.podman_bin -Arguments @("machine", "list")
$steps += $machineList
$machinePresent = ($machineList.stdout_tail -match [regex]::Escape($cfg.machine_name))

if (-not $machinePresent) {
  $steps += Run-Step -Name "podman_machine_init" -FilePath $cfg.podman_bin -Arguments @(
    "machine", "init",
    "--cpus", [string]$cfg.default_cpus,
    "--memory", [string]$cfg.default_memory_mib,
    "--disk-size", [string]$cfg.default_disk_gib,
    "--user-mode-networking",
    "--volume", $cfg.project_mount,
    "--volume", $cfg.workspace_mount,
    $cfg.machine_name
  ) -TimeoutSeconds 600
}

$steps += Run-Step -Name "podman_machine_start" -FilePath $cfg.podman_bin -Arguments @("machine", "start", $cfg.machine_name) -TimeoutSeconds 300
$steps += Run-Step -Name "podman_info" -FilePath $cfg.podman_bin -Arguments @("info", "--format", "json")

if ($RunContainerSmoke) {
  $steps += Run-Step -Name "podman_alpine_smoke" -FilePath $cfg.podman_bin -Arguments @(
    "run", "--rm", "--pull=missing",
    "-v", $cfg.project_mount,
    "-v", $cfg.workspace_mount,
    "--workdir", "/workspace",
    "alpine:3.20",
    "sh", "-lc", "echo theseus-podman-ok && pwd && ls -la | head"
  )
}

$ok = -not ($steps | Where-Object { -not $_.ok })
$needsReboot = [bool]($steps | Where-Object { ($_.stderr_tail -match "reboot") -or ($_.stderr_tail -match "WSL") -or ($_.error -match "WSL") })
$next = @()
if (-not $ok -and $needsReboot) {
  $next += "Reboot Windows, then rerun scripts/setup_podman_sandbox_windows.ps1."
} elseif (-not $ok) {
  $next += "Inspect the failed step stderr in reports/podman_sandbox_setup.json."
} else {
  $next += "Run benchmark adapter smoke for source_terminal_bench and other sandbox-gated cards."
}

$payload = [ordered]@{
  policy = "project_theseus_podman_sandbox_setup_v0"
  created_utc = (Get-Date).ToUniversalTime().ToString("o")
  ok = $ok
  status = if ($ok) { "podman_sandbox_ready" } elseif ($needsReboot) { "blocked_reboot_required_after_wsl_install" } else { "blocked_podman_machine_setup_failed" }
  checks = $checks
  steps = $steps
  safety = [ordered]@{
    network_during_scoring = "forbidden"
    public_benchmark_training_use_allowed = $false
    external_inference_calls = 0
    teacher_apply_mode_allowed = $false
  }
  next_actions = $next
}

Write-JsonFile $Out $payload

$md = @(
  "# Podman Sandbox Setup",
  "",
  ("Created: " + $payload.created_utc),
  "",
  ("- status: " + $payload.status),
  ("- ok: " + $payload.ok),
  ("- podman: " + $cfg.podman_bin),
  ("- machine: " + $cfg.machine_name),
  "",
  "## Next Actions",
  ""
) + ($next | ForEach-Object { "- $_" })

$mdPath = Join-Path $Root $MarkdownOut
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $mdPath) | Out-Null
[System.IO.File]::WriteAllText($mdPath, (($md -join "`n") + "`n"), [System.Text.UTF8Encoding]::new($false))
$payload | ConvertTo-Json -Depth 12
