param(
  [int]$DashboardPort = 8787,
  [string]$DashboardHost = "0.0.0.0",
  [int]$HivePort = 8791,
  [int]$RelayPort = 8793,
  [string]$RelayUrl = "",
  [string]$HiveId = "",
  [string]$HiveSecret = "",
  [switch]$Restart,
  [switch]$StartRelay,
  [switch]$NoDashboard,
  [switch]$NoHive
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Python = Join-Path $Root ".venv-puffer\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) {
    $Python = $cmd.Source
  } else {
    $Python = (Get-Command py -ErrorAction Stop).Source
  }
}

function Get-TheseusServiceProcess {
  param([string]$Pattern)
  Get-CimInstance Win32_Process |
    Where-Object {
      $_.ProcessId -ne $PID -and
      $_.CommandLine -and
      ($_.CommandLine -replace "\\", "/") -match $Pattern
    } |
    Select-Object -First 1
}

if ($Restart) {
  $existing = Get-CimInstance Win32_Process |
    Where-Object {
      $_.ProcessId -ne $PID -and
      $_.CommandLine -match "sparkstream_dashboard\.py|hive_node\.py|hive_relay\.py"
    }
  foreach ($process in $existing) {
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
  }
  Start-Sleep -Milliseconds 500
}

if (-not $env:THESEUS_RUNTIME_ROOT) {
  if (Test-Path "D:\") {
    $env:THESEUS_RUNTIME_ROOT = "D:\ProjectTheseus\runtime"
  } else {
    $env:THESEUS_RUNTIME_ROOT = Join-Path $env:LOCALAPPDATA "ProjectTheseus\runtime"
  }
}
$env:THESEUS_DATA_DIR = Join-Path $env:THESEUS_RUNTIME_ROOT "data"
$env:THESEUS_CACHE_DIR = Join-Path $env:THESEUS_RUNTIME_ROOT "cache"
$env:THESEUS_REPORTS_DIR = Join-Path $env:THESEUS_RUNTIME_ROOT "reports"
$env:THESEUS_CHECKPOINTS_DIR = Join-Path $env:THESEUS_RUNTIME_ROOT "checkpoints"
$env:CARGO_TARGET_DIR = Join-Path $env:THESEUS_RUNTIME_ROOT "cargo-target"
& $Python scripts\runtime_paths.py status --create | Out-Null

if ($HiveSecret) { $env:THESEUS_HIVE_SECRET = $HiveSecret }
if ($HiveId) { $env:THESEUS_HIVE_ID = $HiveId }
if ($RelayUrl) { $env:THESEUS_HIVE_RELAY_URL = $RelayUrl }

if (-not $NoDashboard) {
  $existingDashboardProcess = Get-TheseusServiceProcess "sparkstream_dashboard\.py|hive_operator_dashboard\.py"
  $existingDashboard = Get-NetTCPConnection -LocalPort $DashboardPort -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -First 1
  if (-not $existingDashboardProcess -and -not $existingDashboard) {
    Start-Process -FilePath $Python -ArgumentList @("scripts\sparkstream_dashboard.py", "--host", "$DashboardHost", "--port", "$DashboardPort") -WorkingDirectory $Root -WindowStyle Hidden
  }
}

if (-not $NoHive) {
  $existingHiveProcess = Get-TheseusServiceProcess "hive_node\.py"
  $existingHive = Get-NetTCPConnection -LocalPort $HivePort -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -First 1
  if (-not $existingHiveProcess -and -not $existingHive) {
    $hiveArgs = @("scripts\hive_node.py", "daemon", "--port", "$HivePort")
    if ($RelayUrl) { $hiveArgs += @("--relay-url", "$RelayUrl") }
    if ($HiveId) { $hiveArgs += @("--hive-id", "$HiveId") }
    Start-Process -FilePath $Python -ArgumentList $hiveArgs -WorkingDirectory $Root -WindowStyle Hidden
  }
}

if ($StartRelay) {
  $existingRelayProcess = Get-TheseusServiceProcess "hive_relay\.py"
  $existingRelay = Get-NetTCPConnection -LocalPort $RelayPort -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -First 1
  if (-not $existingRelayProcess -and -not $existingRelay) {
    Start-Process -FilePath $Python -ArgumentList @("scripts\hive_relay.py", "--port", "$RelayPort") -WorkingDirectory $Root -WindowStyle Hidden
  }
}

& $Python scripts\hive_scheduler.py --out reports\hive_scheduler.json | Out-Null
& $Python scripts\update_manager.py check --if-enabled-on-start --respect-interval --out reports\update_checkin.json | Out-Null

$LanIp = (& $Python -c "import sys; sys.path.insert(0, 'scripts'); import hive_node; print(hive_node.find_local_ip())").Trim()
Write-Host "Project Theseus dashboard: http://127.0.0.1:$DashboardPort"
if ($DashboardHost -eq "0.0.0.0" -and $LanIp) {
  Write-Host "Project Theseus LAN dashboard: http://$LanIp`:$DashboardPort"
}
Write-Host "Project Theseus Hive node: http://127.0.0.1:$HivePort/api/hive/status"
if ($LanIp) {
  Write-Host "Project Theseus LAN Hive node: http://$LanIp`:$HivePort/api/hive/status"
  Write-Host "Project Theseus LAN mobile operator: http://$LanIp`:$HivePort/mobile"
}
if ($StartRelay) {
  Write-Host "Project Theseus Hive relay: http://127.0.0.1:$RelayPort/mobile"
}
