param(
  [string]$Profile = "inner_loop",
  [switch]$Execute,
  [switch]$AllowTeacher,
  [switch]$AllowNetworkFetch,
  [switch]$NoTeacher,
  [switch]$NoNetworkFetch,
  [switch]$StartDaemon,
  [switch]$Restart,
  [double]$DurationHours = 0,
  [string]$DashboardHost = "0.0.0.0",
  [int]$Port = 8787
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$TeacherEnabled = -not $NoTeacher
if ($AllowTeacher) { $TeacherEnabled = $true }
$NetworkFetchEnabled = -not $NoNetworkFetch
if ($AllowNetworkFetch) { $NetworkFetchEnabled = $true }

$Python = Join-Path $Root ".venv-puffer\Scripts\python.exe"
if (-not (Test-Path $Python)) {
$cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) {
    $Python = $cmd.Source
  } else {
    $Python = (Get-Command py -ErrorAction Stop).Source
  }
}

function Get-SparkStreamServiceProcess {
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
  $existingSparkStream = Get-CimInstance Win32_Process |
    Where-Object {
      $_.ProcessId -ne $PID -and
      $_.CommandLine -match "sparkstream_dashboard\.py|sparkstream_daemon\.py|autonomy_cycle\.py|run_training_ratchet_profile\.py"
    }
  foreach ($process in $existingSparkStream) {
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
  }
  Start-Sleep -Milliseconds 500
}

$existingDashboardProcess = Get-SparkStreamServiceProcess "sparkstream_dashboard\.py|hive_operator_dashboard\.py"
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -First 1
if ($Restart -and $existing) {
  Stop-Process -Id $existing -Force -ErrorAction SilentlyContinue
  Start-Sleep -Milliseconds 500
  $existing = $null
}
if (-not $existingDashboardProcess -and -not $existing) {
  Start-Process -FilePath $Python -ArgumentList @("scripts\sparkstream_dashboard.py", "--host", "$DashboardHost", "--port", "$Port") -WorkingDirectory $Root -WindowStyle Hidden
}

if ($StartDaemon) {
  $existingDaemons = @(Get-CimInstance Win32_Process |
    Where-Object {
      $_.CommandLine -and ($_.CommandLine -replace "\\", "/") -match "sparkstream_daemon\.py"
    })
  if ($Restart) {
    foreach ($daemon in $existingDaemons) {
      Stop-Process -Id $daemon.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 500
    $existingDaemons = @()
  }
  $args = @("scripts\sparkstream_daemon.py", "--profile", $Profile)
  if ($Execute) { $args += "--execute" }
  if ($TeacherEnabled) { $args += "--allow-teacher" }
  if ($NetworkFetchEnabled) { $args += "--allow-network-fetch" }
  if ($DurationHours -gt 0) { $args += @("--duration-hours", "$DurationHours") }
  if (-not $existingDaemons -or $existingDaemons.Count -eq 0) {
    Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $Root -WindowStyle Hidden
  } else {
    Write-Host "SparkStream daemon already running. Use -Restart to replace it."
  }
}

$LanIp = $null
try {
  $LanIp = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object {
      $_.IPAddress -notlike "127.*" -and
      $_.IPAddress -notlike "169.254.*" -and
      $_.PrefixOrigin -ne "WellKnown"
    } |
    Select-Object -ExpandProperty IPAddress -First 1
} catch {
  $LanIp = $null
}

Write-Host "SparkStream dashboard: http://127.0.0.1:$Port"
if ($DashboardHost -eq "0.0.0.0" -and $LanIp) {
  Write-Host "SparkStream LAN dashboard: http://$LanIp`:$Port"
}
