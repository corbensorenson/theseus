param(
  [string]$TaskName = "Project Theseus Hive Utilization Loop",
  [int]$InitialDelayMinutes = 1,
  [int]$RecoveryIntervalMinutes = 15,
  [int]$SleepSeconds = 60,
  [int]$MaxNewJobs = 3,
  [string]$Profile = "smoke",
  [switch]$Execute,
  [switch]$RunAtStartup,
  [switch]$RunNow,
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
  "scripts\hive_utilization_manager.py",
  "loop",
  "--profile", "$Profile",
  "--max-new-jobs", "$MaxNewJobs",
  "--cycles", "0",
  "--sleep-seconds", "$SleepSeconds",
  "--out", "reports\hive_utilization_manager.json"
)
if ($Execute) { $args += "--execute" }

$argLine = ($args | ForEach-Object {
  if ($_ -match '\s') { '"' + ($_ -replace '"','\"') + '"' } else { $_ }
}) -join " "

$action = New-ScheduledTaskAction -Execute $Python -Argument $argLine -WorkingDirectory $Root
$triggers = @()
$triggers += New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes([Math]::Max(1, $InitialDelayMinutes)) -RepetitionInterval (New-TimeSpan -Minutes ([Math]::Max(5, $RecoveryIntervalMinutes))) -RepetitionDuration (New-TimeSpan -Days 30)
if ($RunAtStartup) {
  $triggers += New-ScheduledTaskTrigger -AtStartup
}
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable `
  -MultipleInstances IgnoreNew `
  -ExecutionTimeLimit (New-TimeSpan -Hours 24)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Settings $settings -Description "Keeps safe Project Theseus Hive CPU/CUDA/MLX slots fed with bounded registered work." -Force | Out-Null

$report = @{
  policy = "project_theseus_hive_utilization_windows_task_v1"
  created_utc = (Get-Date).ToUniversalTime().ToString("o")
  task_name = $TaskName
  initial_delay_minutes = $InitialDelayMinutes
  recovery_interval_minutes = $RecoveryIntervalMinutes
  sleep_seconds = $SleepSeconds
  max_new_jobs = $MaxNewJobs
  profile = $Profile
  execute = [bool]$Execute
  run_at_startup = [bool]$RunAtStartup
  command = "$Python $argLine"
  report = "reports/hive_utilization_manager.json"
  rules = @{
    arbitrary_shell = $false
    public_benchmarks = "calibration_only_not_training"
    teacher = "not_used_by_utilization_loop"
    rented_compute = "not_launched_by_utilization_loop"
  }
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root "reports") | Out-Null
$report | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $Root "reports\hive_utilization_windows_task.json") -Encoding UTF8

if ($RunNow) {
  Start-ScheduledTask -TaskName $TaskName
}

Write-Host "Installed scheduled task: $TaskName"
Write-Host "Report: reports\hive_utilization_windows_task.json"
Write-Host "Latest utilization report: reports\hive_utilization_manager.json"
