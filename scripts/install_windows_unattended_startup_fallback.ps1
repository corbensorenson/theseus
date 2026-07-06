param(
  [int]$VacationDelayMinutes = 180,
  [int]$UtilizationSleepSeconds = 60,
  [int]$MaxNewJobs = 3,
  [string]$Profile = "smoke",
  [int]$VacationSleepSeconds = 1800,
  [int]$ActionTimeoutSeconds = 21600,
  [switch]$Execute,
  [switch]$AllowTeacher,
  [switch]$AllowNetworkFetch,
  [switch]$Explore,
  [switch]$StartServices,
  [switch]$StartNow,
  [switch]$Remove
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Startup = [Environment]::GetFolderPath("Startup")
if (-not $Startup) {
  $Startup = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
}
New-Item -ItemType Directory -Force -Path $Startup | Out-Null

$UtilCmd = Join-Path $Startup "Project Theseus Hive Utilization.cmd"
$VacationCmd = Join-Path $Startup "Project Theseus Vacation Mode.cmd"

if ($Remove) {
  Remove-Item -LiteralPath $UtilCmd -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $VacationCmd -Force -ErrorAction SilentlyContinue
  Write-Host "Removed user-startup fallback launchers."
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

if (Test-Path "D:\") {
  $RuntimeRoot = "D:\ProjectTheseus\runtime"
} else {
  $RuntimeRoot = Join-Path $env:LOCALAPPDATA "ProjectTheseus\runtime"
}

$utilArgs = @(
  "scripts\hive_utilization_manager.py",
  "loop",
  "--profile", "$Profile",
  "--max-new-jobs", "$MaxNewJobs",
  "--cycles", "0",
  "--sleep-seconds", "$UtilizationSleepSeconds",
  "--out", "reports\hive_utilization_manager.json"
)
if ($Execute) { $utilArgs += "--execute" }

$vacArgs = @(
  "scripts\vacation_mode_supervisor.py",
  "--cycles", "0",
  "--sleep-seconds", "$VacationSleepSeconds",
  "--action-timeout-seconds", "$ActionTimeoutSeconds",
  "--teacher-timeout-seconds", "$ActionTimeoutSeconds",
  "--out", "reports\vacation_mode_supervisor.json",
  "--markdown-out", "reports\vacation_mode_supervisor.md"
)
if ($Execute) { $vacArgs += "--execute" }
if ($AllowTeacher) { $vacArgs += "--allow-teacher" }
if ($AllowNetworkFetch) { $vacArgs += "--allow-network-fetch" }
if ($Explore) { $vacArgs += "--explore" }
if ($StartServices) { $vacArgs += "--start-services" }

$utilLine = '"' + $Python + '" ' + (($utilArgs | ForEach-Object { '"' + ($_ -replace '"','\"') + '"' }) -join " ")
$vacPsLine = "& '$($Python -replace "'", "''")' " + (($vacArgs | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join " ")

$utilContent = @"
@echo off
set THESEUS_RUNTIME_ROOT=$RuntimeRoot
cd /d "$Root"
$utilLine
"@

$vacContent = @"
@echo off
set THESEUS_RUNTIME_ROOT=$RuntimeRoot
cd /d "$Root"
powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "Start-Sleep -Seconds $([Math]::Max(0, $VacationDelayMinutes * 60)); Set-Location '$($Root -replace "'", "''")'; `$env:THESEUS_RUNTIME_ROOT='$($RuntimeRoot -replace "'", "''")'; $vacPsLine"
"@

Set-Content -LiteralPath $UtilCmd -Value $utilContent -Encoding ASCII
Set-Content -LiteralPath $VacationCmd -Value $vacContent -Encoding ASCII

if ($StartNow) {
  Start-Process -FilePath $Python -ArgumentList $utilArgs -WorkingDirectory $Root -WindowStyle Hidden
  $delaySeconds = [Math]::Max(0, $VacationDelayMinutes * 60)
  $delayedCommand = "Start-Sleep -Seconds $delaySeconds; Set-Location '$Root'; `$env:THESEUS_RUNTIME_ROOT='$RuntimeRoot'; & '$Python' " + (($vacArgs | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join " ")
  Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-Command", $delayedCommand) -WorkingDirectory $Root -WindowStyle Hidden
}

$report = @{
  policy = "project_theseus_windows_unattended_startup_fallback_v1"
  created_utc = (Get-Date).ToUniversalTime().ToString("o")
  startup_folder = $Startup
  utilization_launcher = $UtilCmd
  vacation_launcher = $VacationCmd
  vacation_delay_minutes = $VacationDelayMinutes
  utilization_sleep_seconds = $UtilizationSleepSeconds
  vacation_sleep_seconds = $VacationSleepSeconds
  max_new_jobs = $MaxNewJobs
  profile = $Profile
  execute = [bool]$Execute
  allow_teacher = [bool]$AllowTeacher
  allow_network_fetch = [bool]$AllowNetworkFetch
  explore = [bool]$Explore
  start_services = [bool]$StartServices
  start_now = [bool]$StartNow
  scheduler_fallback_reason = "Register-ScheduledTask can require elevated permissions on this Windows install."
  reports = @{
    utilization = "reports/hive_utilization_manager.json"
    vacation = "reports/vacation_mode_supervisor.json"
    vacation_markdown = "reports/vacation_mode_supervisor.md"
  }
  safety = @{
    public_benchmarks = "calibration_only_not_training"
    arbitrary_shell = $false
    teacher = "architecture_guidance_and_experiment_specs_only"
  }
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root "reports") | Out-Null
$report | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $Root "reports\windows_unattended_startup_fallback.json") -Encoding UTF8

Write-Host "Installed user-startup fallback launchers."
Write-Host "Utilization: $UtilCmd"
Write-Host "Vacation Mode: $VacationCmd"
Write-Host "Report: reports\windows_unattended_startup_fallback.json"
