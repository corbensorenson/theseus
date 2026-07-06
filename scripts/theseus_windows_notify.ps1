param(
  [string]$Title = "Project Theseus Hive",
  [string]$Message = "Project Theseus notification.",
  [ValidateSet("Info", "Warning", "Error")]
  [string]$Level = "Info",
  [string]$OpenUrl = "",
  [int]$DurationMs = 4000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$IconPath = Join-Path $Root "assets\windows\theseus-hive.ico"

function Escape-ToastText([string]$Text) {
  return [System.Security.SecurityElement]::Escape($Text)
}

function Show-WinRtToast([string]$ToastTitle, [string]$ToastMessage) {
  try {
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
    $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
    $titleText = Escape-ToastText $ToastTitle
    $bodyText = Escape-ToastText $ToastMessage
    $xml.LoadXml("<toast><visual><binding template=`"ToastGeneric`"><text>$titleText</text><text>$bodyText</text></binding></visual></toast>")
    $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Project Theseus Hive").Show($toast)
    return $true
  } catch {
    return $false
  }
}

function Show-TrayBalloon([string]$BalloonTitle, [string]$BalloonMessage, [string]$BalloonLevel) {
  Add-Type -AssemblyName System.Windows.Forms
  Add-Type -AssemblyName System.Drawing
  $notify = New-Object System.Windows.Forms.NotifyIcon
  if (Test-Path $IconPath) {
    $notify.Icon = New-Object System.Drawing.Icon($IconPath)
  } else {
    $notify.Icon = [System.Drawing.SystemIcons]::Application
  }
  $notify.Visible = $true
  $icon = [System.Windows.Forms.ToolTipIcon]::Info
  if ($BalloonLevel -eq "Warning") { $icon = [System.Windows.Forms.ToolTipIcon]::Warning }
  if ($BalloonLevel -eq "Error") { $icon = [System.Windows.Forms.ToolTipIcon]::Error }
  $notify.ShowBalloonTip($DurationMs, $BalloonTitle, $BalloonMessage, $icon)
  Start-Sleep -Milliseconds ([Math]::Max(1200, [Math]::Min($DurationMs, 6000)))
  $notify.Visible = $false
  $notify.Dispose()
}

$shown = Show-WinRtToast $Title $Message
if (-not $shown) {
  Show-TrayBalloon $Title $Message $Level
}
if ($OpenUrl) {
  Start-Process $OpenUrl | Out-Null
}

[ordered]@{
  ok = $true
  policy = "project_theseus_windows_notification_v1"
  created_utc = (Get-Date).ToUniversalTime().ToString("o")
  title = $Title
  level = $Level
  winrt_toast = [bool]$shown
  fallback_balloon = -not [bool]$shown
} | ConvertTo-Json -Depth 4
