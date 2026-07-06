param(
  [int]$IntervalMinutes = 30,
  [int]$ActionTimeoutSeconds = 21600,
  [switch]$Execute,
  [switch]$AllowTeacher,
  [switch]$AllowNetworkFetch,
  [switch]$Explore,
  [switch]$StartServices
)

$ErrorActionPreference = "Continue"
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

New-Item -ItemType Directory -Force -Path (Join-Path $Root "reports") | Out-Null

while ($true) {
  $stopFlags = @(
    (Join-Path $Root "reports\sparkstream_stop.flag"),
    (Join-Path $Root "reports\unattended_autonomy_stop.flag"),
    (Join-Path $Root "reports\vacation_mode_stop.flag")
  )
  if ($stopFlags | Where-Object { Test-Path $_ }) {
    break
  }

  $args = @(
    "scripts\vacation_mode_supervisor.py",
    "--cycles", "1",
    "--sleep-seconds", "1",
    "--action-timeout-seconds", "$ActionTimeoutSeconds",
    "--out", "reports\vacation_mode_supervisor.json",
    "--markdown-out", "reports\vacation_mode_supervisor.md"
  )
  if ($Execute) { $args += "--execute" }
  if ($AllowTeacher) { $args += "--allow-teacher" }
  if ($AllowNetworkFetch) { $args += "--allow-network-fetch" }
  if ($Explore) { $args += "--explore" }
  if ($StartServices) { $args += "--start-services" }

  $started = (Get-Date).ToUniversalTime().ToString("o")
  $loopReport = @{
    policy = "project_theseus_vacation_mode_startup_loop_v1"
    created_utc = $started
    pid = $PID
    interval_minutes = $IntervalMinutes
    execute = [bool]$Execute
    allow_teacher = [bool]$AllowTeacher
    allow_network_fetch = [bool]$AllowNetworkFetch
    explore = [bool]$Explore
    start_services = [bool]$StartServices
    command = "$Python $($args -join ' ')"
  }
  $loopReport | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $Root "reports\vacation_mode_startup_loop.json") -Encoding UTF8

  & $Python @args | Out-Null
  Start-Sleep -Seconds ([Math]::Max(60, $IntervalMinutes * 60))
}
