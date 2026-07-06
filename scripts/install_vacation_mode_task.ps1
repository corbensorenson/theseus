param(
  [string]$TaskName = "Project Theseus Vacation Mode",
  [int]$IntervalMinutes = 30,
  [int]$InitialDelayMinutes = 2,
  [int]$ActionTimeoutSeconds = 21600,
  [switch]$Execute,
  [switch]$AllowTeacher,
  [switch]$AllowNetworkFetch,
  [switch]$Explore,
  [switch]$StartServices,
  [switch]$RunAtStartup,
  [switch]$RunNow,
  [switch]$NoStartupFallback,
  [switch]$Remove
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if ($Remove) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
  Write-Host "Removed scheduled task: $TaskName"
  exit 0
}

$Python = Join-Path $Root ".venv-puffer\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) {
    $Python = $cmd.Source
  } else {
    $Python = (Get-Command py -ErrorAction Stop).Source
  }
}

if (-not $env:THESEUS_RUNTIME_ROOT) {
  if (Test-Path "D:\") {
    $env:THESEUS_RUNTIME_ROOT = "D:\ProjectTheseus\runtime"
  } else {
    $env:THESEUS_RUNTIME_ROOT = Join-Path $env:LOCALAPPDATA "ProjectTheseus\runtime"
  }
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

$argLine = ($args | ForEach-Object {
  if ($_ -match '\s') { '"' + ($_ -replace '"','\"') + '"' } else { $_ }
}) -join " "

$installMode = "scheduled_task"
$installError = ""
$startupFile = ""
try {
  $action = New-ScheduledTaskAction -Execute $Python -Argument $argLine -WorkingDirectory $Root
  $triggers = @()
  $triggers += New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes([Math]::Max(1, $InitialDelayMinutes)) -RepetitionInterval (New-TimeSpan -Minutes ([Math]::Max(5, $IntervalMinutes))) -RepetitionDuration (New-TimeSpan -Days 30)
  if ($RunAtStartup) {
    $triggers += New-ScheduledTaskTrigger -AtStartup
  }
  $settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8)

  Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Settings $settings -Description "Runs Project Theseus Vacation Mode Supervisor V3 with bounded VIEA autonomy, Hive work-board execution, and long-run governor reporting." -Force | Out-Null
} catch {
  $installError = $_.Exception.Message
  $installMode = "startup_loop_fallback"
  if ($NoStartupFallback) {
    throw
  }
  $startupDir = [Environment]::GetFolderPath("Startup")
  New-Item -ItemType Directory -Force -Path $startupDir | Out-Null
  $startupFile = Join-Path $startupDir "Project Theseus Vacation Mode.cmd"
  $loopArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$Root\scripts\vacation_mode_startup_loop.ps1`"",
    "-IntervalMinutes", "$IntervalMinutes",
    "-ActionTimeoutSeconds", "$ActionTimeoutSeconds"
  )
  if ($Execute) { $loopArgs += "-Execute" }
  if ($AllowTeacher) { $loopArgs += "-AllowTeacher" }
  if ($AllowNetworkFetch) { $loopArgs += "-AllowNetworkFetch" }
  if ($Explore) { $loopArgs += "-Explore" }
  if ($StartServices) { $loopArgs += "-StartServices" }
  $loopLine = ($loopArgs | ForEach-Object { $_ }) -join " "
  $cmdText = @"
@echo off
cd /d "$Root"
start "Project Theseus Vacation Mode" /min powershell $loopLine
"@
  $cmdText | Set-Content -Path $startupFile -Encoding ASCII
}

$report = @{
  policy = "project_theseus_vacation_mode_windows_task_v1"
  created_utc = (Get-Date).ToUniversalTime().ToString("o")
  task_name = $TaskName
  interval_minutes = $IntervalMinutes
  initial_delay_minutes = $InitialDelayMinutes
  execute = [bool]$Execute
  allow_teacher = [bool]$AllowTeacher
  allow_network_fetch = [bool]$AllowNetworkFetch
  explore = [bool]$Explore
  start_services = [bool]$StartServices
  run_at_startup = [bool]$RunAtStartup
  install_mode = $installMode
  install_error = $installError
  startup_file = $startupFile
  command = "$Python $argLine"
  report = "reports/vacation_mode_supervisor.json"
  rules = @{
    public_benchmarks = "calibration_only_not_training"
    network_fetch = "small_governed_source_or_metadata_fetch_only_when_enabled"
    teacher = "proposal_only_architecture_guidance"
  }
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root "reports") | Out-Null
$report | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $Root "reports\vacation_mode_windows_task.json") -Encoding UTF8

if ($RunNow -and $installMode -eq "scheduled_task") {
  Start-ScheduledTask -TaskName $TaskName
} elseif ($RunNow -and $installMode -eq "startup_loop_fallback") {
  $runNowArgs = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    "scripts\vacation_mode_startup_loop.ps1",
    "-IntervalMinutes",
    "$IntervalMinutes",
    "-ActionTimeoutSeconds",
    "$ActionTimeoutSeconds"
  )
  if ($Execute) { $runNowArgs += "-Execute" }
  if ($AllowTeacher) { $runNowArgs += "-AllowTeacher" }
  if ($AllowNetworkFetch) { $runNowArgs += "-AllowNetworkFetch" }
  if ($Explore) { $runNowArgs += "-Explore" }
  if ($StartServices) { $runNowArgs += "-StartServices" }
  Start-Process -FilePath "powershell" -ArgumentList $runNowArgs -WorkingDirectory $Root -WindowStyle Hidden
}

Write-Host "Installed Vacation Mode runner: $TaskName ($installMode)"
if ($installError) { Write-Host "Task Scheduler error: $installError" }
if ($startupFile) { Write-Host "Startup fallback: $startupFile" }
Write-Host "Report: reports\vacation_mode_windows_task.json"
Write-Host "Latest supervisor report: reports\vacation_mode_supervisor.json"
