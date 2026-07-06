param(
  [string]$DistDir = "dist\windows",
  [switch]$BuildExe,
  [switch]$Force,
  [switch]$Sign,
  [switch]$SignIfConfigured,
  [string]$SigningCertificatePath = $env:THESEUS_WINDOWS_SIGNING_CERT_PATH,
  [string]$SigningCertificatePassword = $env:THESEUS_WINDOWS_SIGNING_CERT_PASSWORD,
  [string]$SigningCertificateThumbprint = $env:THESEUS_WINDOWS_SIGNING_CERT_THUMBPRINT,
  [string]$TimestampUrl = $(if ($env:THESEUS_WINDOWS_TIMESTAMP_URL) { $env:THESEUS_WINDOWS_TIMESTAMP_URL } else { "http://timestamp.digicert.com" }),
  [string]$SignToolPath = $env:THESEUS_SIGNTOOL_PATH
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$Dist = Join-Path $Root $DistDir
$Bundle = Join-Path $Dist "ProjectTheseusHiveSetup"
$Zip = Join-Path $Dist "ProjectTheseusHiveSetup.zip"
$Exe = Join-Path $Dist "ProjectTheseusHiveSetup.exe"
$Report = Join-Path $Dist "hive-installer-artifacts.json"
$SigningReport = Join-Path $Dist "windows-signing-report.json"

if ((Test-Path $Bundle) -and -not $Force) {
  throw "Output already exists: $Bundle. Re-run with -Force."
}
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
if (Test-Path $Bundle) { Remove-Item -LiteralPath $Bundle -Recurse -Force }
if (Test-Path $Zip) { Remove-Item -LiteralPath $Zip -Force }
if (Test-Path $Exe) { Remove-Item -LiteralPath $Exe -Force }
if (Test-Path $SigningReport) { Remove-Item -LiteralPath $SigningReport -Force }

$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { $python = (Get-Command py -ErrorAction Stop).Source }

if (-not (Test-Path (Join-Path $Root "assets\windows\theseus-hive.ico"))) {
  & $python scripts\generate_theseus_windows_icon.py --out assets\windows\theseus-hive.ico | Out-Host
}

& $python scripts\hive_usb_writer.py write `
  --out $Bundle `
  --hive-mode public `
  --public-mode off `
  --no-zip `
  --force | Out-Host

$cmd = Join-Path $Bundle "ProjectTheseusHiveSetup.cmd"
$ps1 = Join-Path $Bundle "ProjectTheseusHiveSetup.ps1"
@"
@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ProjectTheseusHiveSetup.ps1" %*
if errorlevel 1 pause
"@ | Set-Content -LiteralPath $cmd -Encoding ASCII

@'
param(
  [string]$Invite = "",
  [string]$CoordinatorUrl = "",
  [switch]$StartNow,
  [switch]$NoTray
)

$ErrorActionPreference = "Stop"
$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Inner = Join-Path $BundleRoot "Install Project Theseus Hive.ps1"
if (-not (Test-Path $Inner)) {
  $Zip = Join-Path $BundleRoot "ProjectTheseusHiveSetup.zip"
  if (-not (Test-Path $Zip)) {
    throw "Missing inner installer and payload zip: $Inner"
  }
  $ExtractRoot = Join-Path $env:TEMP ("ProjectTheseusHiveSetup-" + [Guid]::NewGuid().ToString("N"))
  New-Item -ItemType Directory -Force -Path $ExtractRoot | Out-Null
  Expand-Archive -LiteralPath $Zip -DestinationPath $ExtractRoot -Force
  $BundleRoot = $ExtractRoot
  $Inner = Join-Path $BundleRoot "Install Project Theseus Hive.ps1"
  if (-not (Test-Path $Inner)) {
    throw "Payload zip did not contain Install Project Theseus Hive.ps1"
  }
}
$args = @()
if ($Invite) { $args += @("-Invite", $Invite) }
if ($CoordinatorUrl) { $args += @("-CoordinatorUrl", $CoordinatorUrl) }
if ($StartNow) { $args += "-StartNow" }
if ($NoTray) { $args += "-NoTray" }
& powershell -NoProfile -ExecutionPolicy Bypass -File $Inner @args
'@ | Set-Content -LiteralPath $ps1 -Encoding UTF8

Compress-Archive -Path (Join-Path $Bundle "*") -DestinationPath $Zip -Force
Copy-Item -LiteralPath $Zip -Destination (Join-Path $Bundle "ProjectTheseusHiveSetup.zip") -Force

$iexpress = Join-Path $env:WINDIR "System32\iexpress.exe"
$sed = Join-Path $Dist "ProjectTheseusHiveSetup.sed"
if ($BuildExe -and (Test-Path $iexpress)) {
  @"
[Version]
Class=IEXPRESS
SEDVersion=3
[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=0
HideExtractAnimation=1
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=N
InstallPrompt=
DisplayLicense=
FinishMessage=Project Theseus Hive installer launched.
TargetName=$Exe
FriendlyName=Project Theseus Hive Setup
AppLaunched=ProjectTheseusHiveSetup.cmd
PostInstallCmd=<None>
AdminQuietInstCmd=
UserQuietInstCmd=
SourceFiles=SourceFiles
[Strings]
FILE0="ProjectTheseusHiveSetup.cmd"
FILE1="ProjectTheseusHiveSetup.ps1"
FILE2="ProjectTheseusHiveSetup.zip"
[SourceFiles]
SourceFiles0=$Bundle
[SourceFiles0]
%FILE0%=
%FILE1%=
%FILE2%=
"@ | Set-Content -LiteralPath $sed -Encoding ASCII
  & $iexpress /N /Q $sed | Out-Null
}

$signingConfigured = [bool](($SigningCertificatePath -and (Test-Path $SigningCertificatePath)) -or $SigningCertificateThumbprint)
$shouldSign = [bool]($Sign -or ($SignIfConfigured -and $signingConfigured))
if ($shouldSign) {
  $signPaths = @()
  foreach ($path in @($Exe, $ps1)) {
    if (Test-Path $path) { $signPaths += $path }
  }
  $signArgs = @(
    "-Path"
  ) + $signPaths + @(
    "-Out", $SigningReport,
    "-TimestampUrl", $TimestampUrl
  )
  if ($SigningCertificatePath) { $signArgs += @("-CertificatePath", $SigningCertificatePath) }
  if ($SigningCertificatePassword) { $signArgs += @("-CertificatePassword", $SigningCertificatePassword) }
  if ($SigningCertificateThumbprint) { $signArgs += @("-CertificateThumbprint", $SigningCertificateThumbprint) }
  if ($SignToolPath) { $signArgs += @("-SignToolPath", $SignToolPath) }
  if ($Sign) { $signArgs += "-RequireSignature" }
  & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sign_theseus_windows_artifacts.ps1 @signArgs | Out-Host
  if ($LASTEXITCODE -ne 0 -and $Sign) {
    throw "Windows signing failed. See $SigningReport."
  }
}

function Get-Sha256([string]$Path) {
  if (-not (Test-Path $Path)) { return "" }
  return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

$artifacts = @()
foreach ($path in @($Zip, $Exe, $cmd, $ps1)) {
  if (Test-Path $path) {
    $item = Get-Item -LiteralPath $path
    $artifacts += [ordered]@{
      path = ($item.FullName.Replace("\", "/"))
      name = $item.Name
      size_bytes = $item.Length
      sha256 = Get-Sha256 $item.FullName
      modified_utc = $item.LastWriteTimeUtc.ToString("o")
    }
  }
}
$signing = [ordered]@{
  requested = [bool]$Sign
  sign_if_configured = [bool]$SignIfConfigured
  configured = $signingConfigured
  attempted = $shouldSign
  report = ""
}
if (Test-Path $SigningReport) {
  $signing.report = $SigningReport.Replace("\", "/")
  try {
    $signing.details = Get-Content -Raw -LiteralPath $SigningReport | ConvertFrom-Json
  } catch {
    $signing.details = [ordered]@{ ok = $false; error = $_.Exception.Message }
  }
}
$payload = [ordered]@{
  ok = $true
  policy = "project_theseus_windows_installer_artifacts_v1"
  created_utc = (Get-Date).ToUniversalTime().ToString("o")
  platform = "windows"
  artifact_count = $artifacts.Count
  artifacts = $artifacts
  exe_built = (Test-Path $Exe)
  signing = $signing
  exe_note = if (Test-Path $Exe) { "IExpress self-extracting installer created. Use -Sign with a local Authenticode certificate to reduce SmartScreen friction." } else { "Run with -BuildExe on Windows with iexpress.exe available to create the click installer EXE." }
}
$payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Report -Encoding UTF8
New-Item -ItemType Directory -Force -Path (Join-Path $Root "reports") | Out-Null
Copy-Item -LiteralPath $Report -Destination (Join-Path $Root "reports\hive_installer_artifacts_windows.json") -Force

Write-Host "Windows installer folder: $Bundle"
Write-Host "Windows installer zip: $Zip"
if (Test-Path $Exe) { Write-Host "Windows installer exe: $Exe" }
Write-Host "Windows installer artifact manifest: $Report"
