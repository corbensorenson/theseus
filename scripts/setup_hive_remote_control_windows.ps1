param(
  [switch]$EnableRdp,
  [switch]$Execute,
  [string]$RustDeskId = "",
  [string]$Out = "reports\hive_remote_control_windows_setup.json"
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")

function Test-Administrator {
  $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
  $principal = New-Object Security.Principal.WindowsPrincipal($identity)
  return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-RdpState {
  $terminalServer = "HKLM:\SYSTEM\CurrentControlSet\Control\Terminal Server"
  $deny = $null
  if (Test-Path -LiteralPath $terminalServer) {
    $deny = (Get-ItemProperty -LiteralPath $terminalServer -Name fDenyTSConnections -ErrorAction SilentlyContinue).fDenyTSConnections
  }
  $service = Get-Service -Name TermService -ErrorAction SilentlyContinue
  $firewall = @()
  try {
    $firewall = Get-NetFirewallRule -DisplayGroup "Remote Desktop" -ErrorAction Stop |
      Select-Object -First 8 -Property DisplayName,Enabled,Profile
  } catch {
    $firewall = @()
  }
  return [ordered]@{
    host_enabled = ($deny -eq 0)
    deny_connections_value = $deny
    service_status = if ($service) { $service.Status.ToString() } else { "missing" }
    firewall_rules = $firewall
  }
}

function Write-RemoteControlConfig([string]$RustDeskIdValue) {
  if (-not $RustDeskIdValue) { return $null }
  $path = Join-Path $repo "configs\hive_remote_control.local.json"
  $payload = [ordered]@{
    policy = "project_theseus_hive_remote_control_local_v0"
    updated_utc = [DateTimeOffset]::UtcNow.ToString("o")
    providers = [ordered]@{
      rustdesk = [ordered]@{
        id = $RustDeskIdValue
      }
    }
  }
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $path) | Out-Null
  $payload | ConvertTo-Json -Depth 8 | Set-Content -Path $path -Encoding UTF8
  return $path
}

$isAdmin = Test-Administrator
$before = Get-RdpState
$actions = @()

if ($EnableRdp) {
  $actions += "enable_windows_remote_desktop"
  if ($Execute) {
    if (-not $isAdmin) {
      throw "Enabling Windows Remote Desktop requires an elevated PowerShell session."
    }
    Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\Terminal Server" -Name fDenyTSConnections -Value 0
    Start-Service -Name TermService
    Enable-NetFirewallRule -DisplayGroup "Remote Desktop" | Out-Null
  }
}

$configPath = Write-RemoteControlConfig $RustDeskId
$after = Get-RdpState

$report = [ordered]@{
  ok = $true
  policy = "project_theseus_hive_remote_control_windows_setup_v0"
  created_utc = [DateTimeOffset]::UtcNow.ToString("o")
  executed = [bool]$Execute
  is_administrator = $isAdmin
  actions_requested = $actions
  before = $before
  after = $after
  rustdesk_config_written = [bool]$configPath
  rustdesk_config_path = if ($configPath) { $configPath.ToString() } else { "" }
  next_actions = @(
    "Use LAN or WireGuard/private tunnel before connecting with RDP.",
    "Install RustDesk and pass -RustDeskId when you want cross-platform phone handoff metadata.",
    "Run: python scripts\hive_remote_control.py status"
  )
}

$outPath = Join-Path $repo $Out
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outPath) | Out-Null
$report | ConvertTo-Json -Depth 8 | Set-Content -Path $outPath -Encoding UTF8
$report | ConvertTo-Json -Depth 8
