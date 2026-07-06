param(
  [int]$DashboardPort = 8787,
  [int]$HivePort = 8791,
  [int]$RelayPort = 8793,
  [switch]$StartOnLaunch,
  [switch]$StartRelay,
  [int]$PollSeconds = 15,
  [switch]$StatusOnce,
  [switch]$DisableReportNotifications,
  [switch]$TestNotification
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Test-TheseusTcpPort([string]$HostName, [int]$Port) {
  $client = $null
  try {
    $client = New-Object System.Net.Sockets.TcpClient
    $async = $client.BeginConnect($HostName, $Port, $null, $null)
    if (-not $async.AsyncWaitHandle.WaitOne(350, $false)) { return $false }
    $client.EndConnect($async)
    return $true
  } catch {
    return $false
  } finally {
    if ($client) { $client.Close() }
  }
}

function Get-TheseusPython {
  $venv = Join-Path $Root ".venv-puffer\Scripts\python.exe"
  if (Test-Path $venv) { return $venv }
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  return (Get-Command py -ErrorAction Stop).Source
}

function Get-TheseusIconPath {
  $icon = Join-Path $Root "assets\windows\theseus-hive.ico"
  if (Test-Path $icon) { return $icon }
  return ""
}

function Read-TheseusJson([string]$Path) {
  try {
    if (-not (Test-Path $Path)) { return $null }
    return Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
  } catch {
    return $null
  }
}

function Write-TheseusJson([string]$Path, [object]$Value) {
  try {
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Path -Encoding UTF8
  } catch {
  }
}

function Get-TheseusValue([object]$Object, [string]$Name, [object]$Default) {
  if ($null -eq $Object) { return $Default }
  if ($Object -is [System.Collections.IDictionary] -and $Object.Contains($Name)) {
    return $Object[$Name]
  }
  $prop = $Object.PSObject.Properties[$Name]
  if ($prop) { return $prop.Value }
  return $Default
}

function Get-TheseusEventState([object]$Events, [string]$Name) {
  if ($null -eq $Events) { return $null }
  if ($Events -is [System.Collections.IDictionary] -and $Events.Contains($Name)) {
    return $Events[$Name]
  }
  $prop = $Events.PSObject.Properties[$Name]
  if ($prop) { return $prop.Value }
  return $null
}

function Set-TheseusEventState([object]$Events, [string]$Name, [object]$Value) {
  if ($Events -is [System.Collections.IDictionary]) {
    $Events[$Name] = $Value
  } else {
    $Events | Add-Member -NotePropertyName $Name -NotePropertyValue $Value -Force
  }
}

function Get-TheseusStatus {
  [ordered]@{
    ok = $true
    policy = "project_theseus_windows_tray_status_v1"
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
    root = $Root
    dashboard = [ordered]@{
      live = (Test-TheseusTcpPort "127.0.0.1" $DashboardPort)
      url = "http://127.0.0.1:$DashboardPort"
    }
    hive = [ordered]@{
      live = (Test-TheseusTcpPort "127.0.0.1" $HivePort)
      url = "http://127.0.0.1:$HivePort/mobile"
      status_url = "http://127.0.0.1:$HivePort/api/hive/status"
    }
    relay = [ordered]@{
      live = (Test-TheseusTcpPort "127.0.0.1" $RelayPort)
      url = "http://127.0.0.1:$RelayPort/mobile"
    }
  }
}

function Start-TheseusHidden([string[]]$Args) {
  Start-Process -FilePath "powershell.exe" -ArgumentList $Args -WorkingDirectory $Root -WindowStyle Hidden | Out-Null
}

function Start-TheseusServices([switch]$Restart) {
  $args = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $Root "scripts\start_theseus_hive.ps1"),
    "-DashboardPort", "$DashboardPort",
    "-HivePort", "$HivePort",
    "-RelayPort", "$RelayPort"
  )
  if ($Restart) { $args += "-Restart" }
  if ($StartRelay) { $args += "-StartRelay" }
  Start-TheseusHidden $args
}

function Stop-TheseusServices {
  $python = Get-TheseusPython
  Start-Process -FilePath $python -ArgumentList @("scripts\theseus_cli.py", "stop", "--force") -WorkingDirectory $Root -WindowStyle Hidden | Out-Null
}

function Open-TheseusUrl([string]$Url) {
  Start-Process $Url | Out-Null
}

function Open-TheseusPath([string]$Path) {
  Start-Process explorer.exe -ArgumentList "`"$Path`"" | Out-Null
}

function Start-TheseusCudaDoctor {
  $python = Get-TheseusPython
  Start-Process -FilePath $python -ArgumentList @(
    "scripts\windows_cuda_doctor.py",
    "--refresh",
    "--out",
    "reports\windows_cuda_doctor.json",
    "--markdown-out",
    "reports\windows_cuda_doctor.md"
  ) -WorkingDirectory $Root -WindowStyle Hidden | Out-Null
}

function Escape-ToastText([string]$Text) {
  return [System.Security.SecurityElement]::Escape($Text)
}

function Show-TheseusWinRtToast([string]$Title, [string]$Message) {
  try {
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
    $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
    $safeTitle = Escape-ToastText $Title
    $safeMessage = Escape-ToastText $Message
    $xml.LoadXml("<toast><visual><binding template=`"ToastGeneric`"><text>$safeTitle</text><text>$safeMessage</text></binding></visual></toast>")
    $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Project Theseus Hive").Show($toast)
    return $true
  } catch {
    return $false
  }
}

function Show-TheseusNotification([string]$Title, [string]$Message, [string]$Level = "Info") {
  $shown = Show-TheseusWinRtToast $Title $Message
  if ($shown) { return }
  $icon = [System.Windows.Forms.ToolTipIcon]::Info
  if ($Level -eq "Warning") { $icon = [System.Windows.Forms.ToolTipIcon]::Warning }
  if ($Level -eq "Error") { $icon = [System.Windows.Forms.ToolTipIcon]::Error }
  $notify.ShowBalloonTip(4500, $Title, $Message, $icon)
}

function Get-TheseusReportEvents([object]$State) {
  $events = @()
  $broad = Read-TheseusJson (Join-Path $Root "reports\broad_transfer_matrix.json")
  if ($broad -and $broad.summary) {
    $rate = [double](Get-TheseusValue $broad.summary "real_public_pass_rate" 0)
    $tasks = [int](Get-TheseusValue $broad.summary "real_public_task_count" 0)
    $below = @($broad.summary.cards_below_floor).Count
    $fingerprint = "rate=$rate tasks=$tasks below=$below"
    $old = Get-TheseusEventState $State.events "broad_transfer"
    if ($old -and $old.fingerprint -ne $fingerprint) {
      $oldRate = [double](Get-TheseusValue $old "rate" 0)
      if ($rate -gt ($oldRate + 0.000001)) {
        $events += [ordered]@{
          id = "broad_transfer"
          title = "Training improved"
          message = ("Broad transfer rose from {0:N3} to {1:N3} across {2} public tasks." -f $oldRate, $rate, $tasks)
          level = "Info"
        }
      } elseif ($rate -lt ($oldRate - 0.000001)) {
        $events += [ordered]@{
          id = "broad_transfer"
          title = "Training regressed"
          message = ("Broad transfer moved from {0:N3} to {1:N3}; {2} cards remain below floor." -f $oldRate, $rate, $below)
          level = "Warning"
        }
      } elseif ($below -gt 0) {
        $events += [ordered]@{
          id = "broad_transfer"
          title = "Broad transfer updated"
          message = ("Pass rate {0:N3} across {1} tasks; {2} cards remain below floor." -f $rate, $tasks, $below)
          level = "Info"
        }
      }
    }
    Set-TheseusEventState $State.events "broad_transfer" ([ordered]@{ fingerprint = $fingerprint; rate = $rate; tasks = $tasks; below_floor_count = $below })
  }

  $executor = Read-TheseusJson (Join-Path $Root "reports\viea_action_executor.json")
  if ($executor -and $executor.summary) {
    $failed = [int](Get-TheseusValue $executor.summary "failed_total" 0)
    $blocked = [int](Get-TheseusValue $executor.summary "blocked_total" 0)
    $failedThisRun = [int](Get-TheseusValue $executor.summary "failed_this_run" 0)
    $fingerprint = "failed=$failed blocked=$blocked failed_run=$failedThisRun"
    $old = Get-TheseusEventState $State.events "viea_executor"
    if ($old -and $old.fingerprint -ne $fingerprint) {
      $oldFailed = [int](Get-TheseusValue $old "failed_total" 0)
      $oldBlocked = [int](Get-TheseusValue $old "blocked_total" 0)
      if ($failedThisRun -gt 0 -or $failed -gt $oldFailed -or $blocked -gt $oldBlocked) {
        $events += [ordered]@{
          id = "viea_executor"
          title = "Action blocked or failed"
          message = ("VIEA executor now has {0} failed and {1} blocked actions. Open the tray operator for details." -f $failed, $blocked)
          level = "Warning"
        }
      }
    }
    Set-TheseusEventState $State.events "viea_executor" ([ordered]@{ fingerprint = $fingerprint; failed_total = $failed; blocked_total = $blocked })
  }

  $watchdog = Read-TheseusJson (Join-Path $Root "reports\autonomy_watchdog.json")
  if ($watchdog -and $watchdog.summary) {
    $teacherBlocks = [int](Get-TheseusValue $watchdog.summary "teacher_blocks_since_completed" 0)
    $activeWall = [bool](Get-TheseusValue $watchdog.summary "active_frontier_wall" $false)
    $teacherStatus = [string](Get-TheseusValue $watchdog.summary "teacher_last_status" "")
    $fingerprint = "blocks=$teacherBlocks wall=$activeWall status=$teacherStatus"
    $old = Get-TheseusEventState $State.events "teacher_need"
    if ($old -and $old.fingerprint -ne $fingerprint -and $activeWall -and $teacherBlocks -gt 0) {
      $events += [ordered]@{
        id = "teacher_need"
        title = "Teacher architecture help needed"
        message = ("Frontier wall is active and {0} teacher block(s) happened since the last completed result." -f $teacherBlocks)
        level = "Warning"
      }
    }
    Set-TheseusEventState $State.events "teacher_need" ([ordered]@{ fingerprint = $fingerprint; teacher_blocks_since_completed = $teacherBlocks; active_wall = $activeWall })
  }

  $candidate = Read-TheseusJson (Join-Path $Root "reports\candidate_promotion_gate.json")
  if ($candidate) {
    $promote = [bool](Get-TheseusValue $candidate "promote" $false)
    $passed = [int](Get-TheseusValue $candidate "passed" 0)
    $total = [int](Get-TheseusValue $candidate "total" 0)
    $fingerprint = "promote=$promote passed=$passed total=$total"
    $old = Get-TheseusEventState $State.events "candidate_gate"
    if ($old -and $old.fingerprint -ne $fingerprint -and $promote) {
      $events += [ordered]@{
        id = "candidate_gate"
        title = "Candidate promotion ready"
        message = ("Promotion gate passed {0}/{1} checks." -f $passed, $total)
        level = "Info"
      }
    }
    Set-TheseusEventState $State.events "candidate_gate" ([ordered]@{ fingerprint = $fingerprint; promote = $promote; passed = $passed; total = $total })
  }

  $resource = Read-TheseusJson (Join-Path $Root "reports\resource_governor.json")
  if ($resource -and $resource.decision) {
    $canRunRaw = Get-TheseusValue $resource.decision "can_run_requested_profile" $null
    $canRunText = if ($null -eq $canRunRaw) { "unknown" } else { [string]$canRunRaw }
    $reasons = @(Get-TheseusValue $resource.decision "throttle_reasons" @())
    $warnings = @(Get-TheseusValue $resource.decision "warnings" @())
    $gpu = Get-TheseusValue (Get-TheseusValue $resource.current_resources "gpu" $null) "name" ""
    $freeVram = [double](Get-TheseusValue (Get-TheseusValue $resource.current_resources "gpu" $null) "memory_free_mib" 0)
    $tempC = [double](Get-TheseusValue (Get-TheseusValue $resource.current_resources "gpu" $null) "temperature_c" 0)
    $fingerprint = "can=$canRunText reasons=$($reasons -join ',') warnings=$($warnings.Count) free=$([int]$freeVram) temp=$([int]$tempC)"
    $old = Get-TheseusEventState $State.events "windows_cuda_resource"
    if ($old -and $old.fingerprint -ne $fingerprint) {
      if ($canRunRaw -eq $false -or $reasons.Count -gt 0) {
        $reasonText = if ($reasons.Count -gt 0) { $reasons -join ", " } else { "resource governor blocked requested profile" }
        $events += [ordered]@{
          id = "windows_cuda_resource"
          title = "Windows/CUDA throttled"
          message = ("{0}; free VRAM {1:N0} MiB, temp {2:N0} C." -f $reasonText, $freeVram, $tempC)
          level = "Warning"
        }
      } elseif ((Get-TheseusValue $old "can_run" "") -eq "False" -and $canRunRaw -eq $true) {
        $events += [ordered]@{
          id = "windows_cuda_resource"
          title = "Windows/CUDA clear"
          message = ("{0} can run the requested profile again; free VRAM {1:N0} MiB." -f $gpu, $freeVram)
          level = "Info"
        }
      } elseif ($tempC -ge 82) {
        $events += [ordered]@{
          id = "windows_cuda_resource"
          title = "GPU temperature high"
          message = ("{0} is at {1:N0} C. Prefer bounded smoke work until cooling improves." -f $gpu, $tempC)
          level = "Warning"
        }
      }
    }
    Set-TheseusEventState $State.events "windows_cuda_resource" ([ordered]@{
      fingerprint = $fingerprint
      can_run = $canRunText
      free_vram_mib = $freeVram
      temperature_c = $tempC
      throttle_reasons = $reasons
    })
  }
  return $events
}

function Update-TheseusReportNotifications {
  if ($DisableReportNotifications) { return }
  $statePath = Join-Path $Root "reports\windows_tray_notifications.local.json"
  $state = Read-TheseusJson $statePath
  if (-not $state) {
    $state = [pscustomobject][ordered]@{
      policy = "project_theseus_windows_tray_notification_state_v1"
      events = [pscustomobject][ordered]@{}
    }
  }
  $eventsObject = Get-TheseusValue $state "events" $null
  if (-not $eventsObject) {
    $state | Add-Member -NotePropertyName events -NotePropertyValue ([pscustomobject][ordered]@{}) -Force
  }
  $events = Get-TheseusReportEvents $state
  foreach ($event in $events) {
    Show-TheseusNotification $event.title $event.message $event.level
  }
  $state | Add-Member -NotePropertyName updated_utc -NotePropertyValue ((Get-Date).ToUniversalTime().ToString("o")) -Force
  Write-TheseusJson $statePath $state
}

if ($StatusOnce) {
  Get-TheseusStatus | ConvertTo-Json -Depth 6
  exit 0
}

if ($PSVersionTable.PSEdition -eq "Core" -and -not $IsWindows) {
  throw "Project Theseus tray is Windows-only."
}

$mutexName = "Global\ProjectTheseusHiveTray"
$createdNew = $false
$mutex = New-Object System.Threading.Mutex($true, $mutexName, [ref]$createdNew)
if (-not $createdNew) {
  Open-TheseusUrl "http://127.0.0.1:$HivePort/mobile"
  exit 0
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

if ($StartOnLaunch) {
  Start-TheseusServices
}

$notify = New-Object System.Windows.Forms.NotifyIcon
$iconPath = Get-TheseusIconPath
if ($iconPath) {
  $notify.Icon = New-Object System.Drawing.Icon($iconPath)
} else {
  $notify.Icon = [System.Drawing.SystemIcons]::Application
}
$notify.Visible = $true
$notify.Text = "Project Theseus Hive"

if ($TestNotification) {
  Show-TheseusNotification "Project Theseus Hive" "Windows notifications are wired." "Info"
}

$menu = New-Object System.Windows.Forms.ContextMenuStrip
$statusItem = $menu.Items.Add("Checking Project Theseus...")
$statusItem.Enabled = $false
$menu.Items.Add("-") | Out-Null

$openOperator = $menu.Items.Add("Open Hive Operator / Chat")
$openOperator.Add_Click({ Open-TheseusUrl "http://127.0.0.1:$HivePort/mobile" })

$openDashboard = $menu.Items.Add("Open Dashboard")
$openDashboard.Add_Click({ Open-TheseusUrl "http://127.0.0.1:$DashboardPort" })

$openStatus = $menu.Items.Add("Open Hive Status JSON")
$openStatus.Add_Click({ Open-TheseusUrl "http://127.0.0.1:$HivePort/api/hive/status" })

$menu.Items.Add("-") | Out-Null
$startItem = $menu.Items.Add("Start Services")
$startItem.Add_Click({
  Start-TheseusServices
  Show-TheseusNotification "Project Theseus Hive" "Starting dashboard and Hive node." "Info"
})

$restartItem = $menu.Items.Add("Restart Services")
$restartItem.Add_Click({
  Start-TheseusServices -Restart
  Show-TheseusNotification "Project Theseus Hive" "Restarting dashboard and Hive node." "Info"
})

$stopItem = $menu.Items.Add("Stop Services")
$stopItem.Add_Click({
  Stop-TheseusServices
  Show-TheseusNotification "Project Theseus Hive" "Stopping local Theseus helper services." "Warning"
})

$menu.Items.Add("-") | Out-Null
$setupItem = $menu.Items.Add("Open Setup Wizard")
$setupItem.Add_Click({
  $python = Get-TheseusPython
  Start-Process -FilePath $python -ArgumentList @("scripts\theseus_setup_wizard.py", "--open") -WorkingDirectory $Root -WindowStyle Hidden | Out-Null
})

$reportsItem = $menu.Items.Add("Open Reports Folder")
$reportsItem.Add_Click({ Open-TheseusPath (Join-Path $Root "reports") })

$cudaDoctorItem = $menu.Items.Add("Run Windows/CUDA Doctor")
$cudaDoctorItem.Add_Click({
  Start-TheseusCudaDoctor
  Show-TheseusNotification "Windows/CUDA Doctor" "Refreshing CUDA, resource, scheduler, and performance reports." "Info"
})

$cudaDoctorReportItem = $menu.Items.Add("Open Windows/CUDA Doctor Report")
$cudaDoctorReportItem.Add_Click({
  $report = Join-Path $Root "reports\windows_cuda_doctor.md"
  if (Test-Path $report) {
    Start-Process $report | Out-Null
  } else {
    Open-TheseusPath (Join-Path $Root "reports")
  }
})

$projectItem = $menu.Items.Add("Open Project Folder")
$projectItem.Add_Click({ Open-TheseusPath $Root })

$menu.Items.Add("-") | Out-Null
$quitItem = $menu.Items.Add("Exit Tray")
$quitItem.Add_Click({
  $notify.Visible = $false
  $notify.Dispose()
  [System.Windows.Forms.Application]::Exit()
})

$notify.ContextMenuStrip = $menu
$notify.Add_DoubleClick({ Open-TheseusUrl "http://127.0.0.1:$HivePort/mobile" })

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = [Math]::Max(5, $PollSeconds) * 1000
$timer.Add_Tick({
  $status = Get-TheseusStatus
  $dashboard = if ($status.dashboard.live) { "up" } else { "down" }
  $hive = if ($status.hive.live) { "up" } else { "down" }
  $relay = if ($status.relay.live) { "up" } else { "down" }
  $statusItem.Text = "Dashboard $dashboard | Hive $hive | Relay $relay"
  $notify.Text = "Theseus: dashboard $dashboard, hive $hive"
  Update-TheseusReportNotifications
})
$timer.Start()

Show-TheseusNotification "Project Theseus Hive" "Tray operator is running. Double-click to open Hive chat." "Info"
[System.Windows.Forms.Application]::Run()

if ($mutex) {
  $mutex.ReleaseMutex() | Out-Null
  $mutex.Dispose()
}
