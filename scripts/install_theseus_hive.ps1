param(
  [switch]$StartNow,
  [switch]$StartRelay,
  [switch]$CreateDesktopShortcut,
  [int]$DashboardPort = 8787,
  [int]$HivePort = 8791,
  [int]$RelayPort = 8793,
  [string]$Invite = "",
  [string]$RelayUrl = "",
  [string]$CoordinatorUrl = "",
  [string]$HiveId = "",
  [string]$HiveSecret = "",
  [string]$RuntimeRoot = "",
  [switch]$InstallScheduledTask,
  [switch]$InstallTray,
  [switch]$StartTray,
  [switch]$AutoUpdateSoft,
  [switch]$VacationModeTask,
  [switch]$SkipRegistration,
  [string]$PublicMode = "off",
  [string]$PublicGatewayUrl = "",
  [string]$PublicWorkerName = "",
  [switch]$AllowPublic
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$VenvPython = Join-Path $Root ".venv-puffer\Scripts\python.exe"
$Python = $VenvPython
if (-not (Test-Path $Python)) {
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) {
    $Python = $cmd.Source
  } else {
    $Python = (Get-Command py -ErrorAction Stop).Source
  }
}

if (-not (Test-Path $VenvPython)) {
  & $Python -m venv (Join-Path $Root ".venv-puffer")
  $Python = $VenvPython
  & $Python -m pip install --upgrade pip wheel setuptools | Out-Null
  & $Python -m pip install numpy | Out-Null
} else {
  $Python = $VenvPython
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root "reports") | Out-Null
if ($RuntimeRoot) {
  $env:THESEUS_RUNTIME_ROOT = $RuntimeRoot
}
if (-not $env:THESEUS_RUNTIME_ROOT) {
  if (Test-Path "D:\") {
    $env:THESEUS_RUNTIME_ROOT = "D:\ProjectTheseus\runtime"
  } else {
    $env:THESEUS_RUNTIME_ROOT = Join-Path $env:LOCALAPPDATA "ProjectTheseus\runtime"
  }
}
& $Python scripts\runtime_paths.py init --runtime-root $env:THESEUS_RUNTIME_ROOT | Out-Null

if ($AutoUpdateSoft) {
  & $Python scripts\update_manager.py configure --mode auto_soft --check-on-start --auto-install-soft --no-auto-install-hard --out reports\update_client_configure_install.json | Out-Null
  & powershell -ExecutionPolicy Bypass -File scripts\install_hive_update_task.ps1 -RunAtStartup | Out-Null
}

if ($Invite) {
  & $Python scripts\hive_invite.py apply --invite $Invite --write-local-config --out reports\hive_join_apply_last.json | Out-Null
}
if ($RelayUrl -or $CoordinatorUrl -or $HiveId -or $HiveSecret) {
  $configureArgs = @("scripts\hive_invite.py", "configure-local", "--out", "reports\hive_join_configure_last.json")
  if ($RelayUrl) { $configureArgs += @("--relay-url", $RelayUrl) }
  if ($CoordinatorUrl) { $configureArgs += @("--coordinator-url", $CoordinatorUrl) }
  if ($HiveId) { $configureArgs += @("--hive-id", $HiveId) }
  if ($HiveSecret) { $configureArgs += @("--join-token", $HiveSecret) }
  & $Python @configureArgs | Out-Null
}
if (-not $SkipRegistration -and -not (Test-Path (Join-Path $Root "configs\theseus_registration.local.json"))) {
  & $Python scripts\license_manager.py register --usage personal_homelab --accept-terms --out reports\license_registration_windows_install.json | Out-Null
}
if ($PublicMode -ne "off" -or $PublicGatewayUrl) {
  $publicArgs = @(
    "scripts\public_hive_contributor.py",
    "configure",
    "--mode", $PublicMode,
    "--gateway-url", $PublicGatewayUrl,
    "--worker-name", $PublicWorkerName,
    "--out", "reports\public_hive_contribution_status.json"
  )
  if ($AllowPublic) { $publicArgs += "--allow" }
  & $Python @publicArgs | Out-Null
}

& $Python scripts\hive_node.py probe --out reports\hive_status.json | Out-Null
& $Python scripts\hive_scheduler.py --out reports\hive_scheduler.json | Out-Null
$TheseusIcon = Join-Path $Root "assets\windows\theseus-hive.ico"
$TheseusIconLocation = if (Test-Path $TheseusIcon) { $TheseusIcon } else { "powershell.exe,0" }

if ($InstallScheduledTask) {
  $taskName = "Project Theseus Hive"
  $taskAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$Root\scripts\start_theseus_hive.ps1`" -DashboardPort $DashboardPort -HivePort $HivePort -RelayPort $RelayPort" -WorkingDirectory $Root
  $taskTrigger = New-ScheduledTaskTrigger -AtStartup
  $taskSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew
  Register-ScheduledTask -TaskName $taskName -Action $taskAction -Trigger $taskTrigger -Settings $taskSettings -Description "Starts Project Theseus Hive node and dashboard." -Force | Out-Null
}

function New-TheseusShortcut {
  param(
    [string]$Path,
    [string]$TargetPath,
    [string]$Arguments,
    [string]$WorkingDirectory,
    [string]$IconLocation = ""
  )
  $shell = New-Object -ComObject WScript.Shell
  $shortcut = $shell.CreateShortcut($Path)
  $shortcut.TargetPath = $TargetPath
  $shortcut.Arguments = $Arguments
  $shortcut.WorkingDirectory = $WorkingDirectory
  if ($IconLocation) { $shortcut.IconLocation = $IconLocation }
  $shortcut.Save()
}

if ($InstallTray) {
  $programs = [Environment]::GetFolderPath("Programs")
  $startup = [Environment]::GetFolderPath("Startup")
  $startMenuDir = Join-Path $programs "Project Theseus"
  New-Item -ItemType Directory -Force -Path $startMenuDir | Out-Null
  New-Item -ItemType Directory -Force -Path $startup | Out-Null

  $trayArgs = @(
    "-NoProfile",
    "-ExecutionPolicy Bypass",
    "-WindowStyle Hidden",
    "-File `"$Root\scripts\theseus_tray.ps1`"",
    "-StartOnLaunch",
    "-DashboardPort $DashboardPort",
    "-HivePort $HivePort",
    "-RelayPort $RelayPort"
  )
  if ($StartRelay) { $trayArgs += "-StartRelay" }
  $trayArguments = $trayArgs -join " "

  New-TheseusShortcut `
    -Path (Join-Path $startup "Project Theseus Hive Tray.lnk") `
    -TargetPath "powershell.exe" `
    -Arguments $trayArguments `
    -WorkingDirectory $Root `
    -IconLocation $TheseusIconLocation

  New-TheseusShortcut `
    -Path (Join-Path $startMenuDir "Project Theseus Hive Tray.lnk") `
    -TargetPath "powershell.exe" `
    -Arguments $trayArguments `
    -WorkingDirectory $Root `
    -IconLocation $TheseusIconLocation

  New-TheseusShortcut `
    -Path (Join-Path $startMenuDir "Project Theseus Hive Operator.lnk") `
    -TargetPath "cmd.exe" `
    -Arguments "/c start `"`" `"http://127.0.0.1:$HivePort/mobile`"" `
    -WorkingDirectory $Root `
    -IconLocation $TheseusIconLocation

  New-TheseusShortcut `
    -Path (Join-Path $startMenuDir "Project Theseus Dashboard.lnk") `
    -TargetPath "cmd.exe" `
    -Arguments "/c start `"`" `"http://127.0.0.1:$DashboardPort`"" `
    -WorkingDirectory $Root `
    -IconLocation $TheseusIconLocation
}

if ($VacationModeTask) {
  $vacationArgs = @("-ExecutionPolicy", "Bypass", "-File", "scripts\install_vacation_mode_task.ps1", "-Execute", "-StartServices", "-RunAtStartup")
  if ($AutoUpdateSoft) { $vacationArgs += "-Explore" }
  & powershell @vacationArgs | Out-Null
}

if ($CreateDesktopShortcut) {
  $desktop = [Environment]::GetFolderPath("Desktop")
  $setupShortcutPath = Join-Path $desktop "Project Theseus Setup.lnk"
  $setupShell = New-Object -ComObject WScript.Shell
  $setupShortcut = $setupShell.CreateShortcut($setupShortcutPath)
  $setupShortcut.TargetPath = "cmd.exe"
  $setupShortcut.Arguments = "/c `"$Root\bin\project-theseus-setup.cmd`""
  $setupShortcut.WorkingDirectory = $Root
  $setupShortcut.IconLocation = $TheseusIconLocation
  $setupShortcut.Save()

  $shortcutPath = Join-Path $desktop "Project Theseus Hive.lnk"
  $shell = New-Object -ComObject WScript.Shell
  $shortcut = $shell.CreateShortcut($shortcutPath)
  $shortcut.TargetPath = "powershell.exe"
  $startArgs = @(
    "-ExecutionPolicy Bypass",
    "-File `"$Root\scripts\start_theseus_hive.ps1`"",
    "-DashboardPort $DashboardPort",
    "-HivePort $HivePort",
    "-RelayPort $RelayPort"
  )
  if ($StartRelay) { $startArgs += "-StartRelay" }
  if ($RelayUrl) { $startArgs += "-RelayUrl `"$RelayUrl`"" }
  if ($HiveId) { $startArgs += "-HiveId `"$HiveId`"" }
  if ($HiveSecret) { $startArgs += "-HiveSecret `"$HiveSecret`"" }
  $shortcut.Arguments = $startArgs -join " "
  $shortcut.WorkingDirectory = $Root
  $shortcut.IconLocation = $TheseusIconLocation
  $shortcut.Save()
}

if ($StartNow) {
  $startArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts\start_theseus_hive.ps1",
    "-DashboardPort", "$DashboardPort",
    "-HivePort", "$HivePort",
    "-RelayPort", "$RelayPort"
  )
  if ($StartRelay) { $startArgs += "-StartRelay" }
  if ($RelayUrl) { $startArgs += @("-RelayUrl", "$RelayUrl") }
  if ($CoordinatorUrl) { $env:THESEUS_HIVE_COORDINATOR_URL = $CoordinatorUrl }
  if ($HiveId) { $startArgs += @("-HiveId", "$HiveId") }
  if ($HiveSecret) { $startArgs += @("-HiveSecret", "$HiveSecret") }
  & powershell @startArgs
}

if ($StartTray) {
  $trayStartArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-WindowStyle", "Hidden",
    "-File", "scripts\theseus_tray.ps1",
    "-DashboardPort", "$DashboardPort",
    "-HivePort", "$HivePort",
    "-RelayPort", "$RelayPort"
  )
  if ($StartRelay) { $trayStartArgs += "-StartRelay" }
  if ($StartNow) { $trayStartArgs += "-StartOnLaunch" }
  Start-Process -FilePath "powershell.exe" -ArgumentList $trayStartArgs -WorkingDirectory $Root -WindowStyle Hidden | Out-Null
}

Write-Host "Project Theseus Hive installed for this checkout."
if ($Invite) {
  Write-Host "Invite applied to ignored local join config."
}
Write-Host "Run: powershell -ExecutionPolicy Bypass -File scripts\start_theseus_hive.ps1"
if ($InstallScheduledTask) {
  Write-Host "Scheduled task installed: Project Theseus Hive"
}
if ($InstallTray) {
  Write-Host "Tray autostart and Start Menu shortcuts installed: Project Theseus Hive Tray"
}
if ($AutoUpdateSoft) {
  Write-Host "Soft auto-update task installed: Project Theseus Hive Update Check"
}
if ($VacationModeTask) {
  Write-Host "Vacation Mode scheduled task installed."
}
