param(
  [string]$TaskName = "Project Theseus Hive Update Check",
  [int]$IntervalMinutes = 30,
  [switch]$RunAtStartup,
  [switch]$RunNow,
  [switch]$Remove
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")

if ($Remove) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
  Write-Host "Removed scheduled task: $TaskName"
  exit 0
}

$VenvPython = Join-Path $Root ".venv-puffer\Scripts\python.exe"
if (Test-Path $VenvPython) {
  $Python = $VenvPython
} else {
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) {
    $Python = $cmd.Source
  } else {
    $Python = (Get-Command py -ErrorAction Stop).Source
  }
}

$script = @"
Set-Location '$($Root.Path.Replace("'", "''"))'
& '$($Python.Replace("'", "''"))' scripts\update_manager.py check --apply --respect-interval --out reports\update_checkin.json
& '$($Python.Replace("'", "''"))' scripts\hive_version_manager.py status --out reports\hive_version_status.json
"@

$runner = Join-Path $Root "updates\hive_update_check.ps1"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $runner) | Out-Null
$script | Set-Content -LiteralPath $runner -Encoding UTF8

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`"" -WorkingDirectory $Root
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes ([Math]::Max(5, $IntervalMinutes))) -RepetitionDuration ([TimeSpan]::MaxValue)
$triggers = @($trigger)
if ($RunAtStartup) {
  $triggers += New-ScheduledTaskTrigger -AtStartup
}
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Settings $settings -Description "Checks the Project Theseus private Hive catalog and installs safe soft updates." -Force | Out-Null

$report = [ordered]@{
  ok = $true
  policy = "project_theseus_hive_update_task_install_v1"
  created_utc = (Get-Date).ToUniversalTime().ToString("o")
  task_name = $TaskName
  interval_minutes = $IntervalMinutes
  run_at_startup = [bool]$RunAtStartup
  runner = "updates\hive_update_check.ps1"
  python = $Python
  report = "reports\hive_version_status.json"
}
New-Item -ItemType Directory -Force -Path (Join-Path $Root "reports") | Out-Null
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Root "reports\hive_update_task_install.json") -Encoding UTF8

if ($RunNow) {
  Start-ScheduledTask -TaskName $TaskName
}

Write-Host "Installed scheduled task: $TaskName"
Write-Host "Latest status report: reports\hive_version_status.json"
