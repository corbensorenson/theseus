param(
  [string[]]$Path = @(),
  [string]$CertificatePath = $env:THESEUS_WINDOWS_SIGNING_CERT_PATH,
  [string]$CertificatePassword = $env:THESEUS_WINDOWS_SIGNING_CERT_PASSWORD,
  [string]$CertificateThumbprint = $env:THESEUS_WINDOWS_SIGNING_CERT_THUMBPRINT,
  [string]$TimestampUrl = $(if ($env:THESEUS_WINDOWS_TIMESTAMP_URL) { $env:THESEUS_WINDOWS_TIMESTAMP_URL } else { "http://timestamp.digicert.com" }),
  [string]$SignToolPath = $env:THESEUS_SIGNTOOL_PATH,
  [string]$Out = "reports\windows_signing_report.json",
  [switch]$RequireSignature
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Resolve-TheseusPath([string]$Value) {
  if ([System.IO.Path]::IsPathRooted($Value)) { return $Value }
  return (Join-Path $Root $Value)
}

function Find-SignTool {
  if ($SignToolPath -and (Test-Path $SignToolPath)) { return (Resolve-Path $SignToolPath).Path }
  $cmd = Get-Command signtool.exe -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $kits = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
  if (Test-Path $kits) {
    $candidate = Get-ChildItem -LiteralPath $kits -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -match "\\x64\\signtool\.exe$" } |
      Sort-Object FullName -Descending |
      Select-Object -First 1
    if ($candidate) { return $candidate.FullName }
  }
  return ""
}

function Resolve-Certificate {
  if ($CertificateThumbprint) {
    $thumb = $CertificateThumbprint.Replace(" ", "").ToUpperInvariant()
    foreach ($store in @("Cert:\CurrentUser\My", "Cert:\LocalMachine\My")) {
      try {
        $cert = Get-ChildItem -Path $store -ErrorAction SilentlyContinue |
          Where-Object { $_.Thumbprint -and $_.Thumbprint.Replace(" ", "").ToUpperInvariant() -eq $thumb } |
          Select-Object -First 1
        if ($cert) { return $cert }
      } catch {
      }
    }
  }
  if ($CertificatePath -and (Test-Path $CertificatePath)) {
    $flags = [System.Security.Cryptography.X509Certificates.X509KeyStorageFlags]::Exportable -bor
      [System.Security.Cryptography.X509Certificates.X509KeyStorageFlags]::PersistKeySet
    return New-Object System.Security.Cryptography.X509Certificates.X509Certificate2($CertificatePath, $CertificatePassword, $flags)
  }
  return $null
}

function Sign-WithSignTool([string]$File, [string]$Tool) {
  $args = @("sign", "/fd", "SHA256")
  if ($TimestampUrl) { $args += @("/tr", $TimestampUrl, "/td", "SHA256") }
  if ($CertificatePath) {
    $args += @("/f", (Resolve-TheseusPath $CertificatePath))
    if ($CertificatePassword) { $args += @("/p", $CertificatePassword) }
  } elseif ($CertificateThumbprint) {
    $args += @("/sha1", $CertificateThumbprint)
  } else {
    return [ordered]@{ ok = $false; method = "signtool"; error = "certificate_required" }
  }
  $args += $File
  $proc = Start-Process -FilePath $Tool -ArgumentList $args -Wait -NoNewWindow -PassThru
  return [ordered]@{ ok = ($proc.ExitCode -eq 0); method = "signtool"; exit_code = $proc.ExitCode }
}

function Sign-WithPowerShell([string]$File, [object]$Cert) {
  if (-not $Cert) {
    return [ordered]@{ ok = $false; method = "set_authenticode_signature"; error = "certificate_required" }
  }
  $args = @{ FilePath = $File; Certificate = $Cert; HashAlgorithm = "SHA256" }
  if ($TimestampUrl) { $args.TimestampServer = $TimestampUrl }
  $sig = Set-AuthenticodeSignature @args
  return [ordered]@{ ok = ($sig.Status -eq "Valid"); method = "set_authenticode_signature"; status = "$($sig.Status)"; status_message = "$($sig.StatusMessage)" }
}

function Signature-Status([string]$File) {
  try {
    $sig = Get-AuthenticodeSignature -FilePath $File
    $subject = ""
    $thumb = ""
    if ($sig.SignerCertificate) {
      $subject = $sig.SignerCertificate.Subject
      $thumb = $sig.SignerCertificate.Thumbprint
    }
    return [ordered]@{
      status = "$($sig.Status)"
      status_message = "$($sig.StatusMessage)"
      signer_subject = $subject
      signer_thumbprint = $thumb
    }
  } catch {
    return [ordered]@{ status = "Unknown"; status_message = $_.Exception.Message }
  }
}

$resolved = @()
foreach ($item in $Path) {
  if (-not $item) { continue }
  $file = Resolve-TheseusPath $item
  if (Test-Path $file) { $resolved += (Resolve-Path $file).Path }
}
$resolved = @($resolved | Select-Object -Unique)

$signTool = Find-SignTool
$cert = Resolve-Certificate
$configured = [bool](($CertificatePath -and (Test-Path $CertificatePath)) -or $CertificateThumbprint)
$rows = @()

foreach ($file in $resolved) {
  $ext = [System.IO.Path]::GetExtension($file).ToLowerInvariant()
  $row = [ordered]@{
    path = $file.Replace("\", "/")
    extension = $ext
    attempted = $false
    ok = $false
    result = $null
    signature = $null
  }
  if (-not $configured) {
    $row.result = [ordered]@{ ok = $false; error = "signing_certificate_not_configured" }
    $row.ok = -not [bool]$RequireSignature
  } elseif ($ext -eq ".exe") {
    $row.attempted = $true
    if ($signTool) {
      $row.result = Sign-WithSignTool $file $signTool
      $row.ok = [bool]$row.result.ok
    } else {
      $row.result = [ordered]@{ ok = $false; error = "signtool_not_found" }
    }
  } elseif ($ext -eq ".ps1") {
    $row.attempted = $true
    $row.result = Sign-WithPowerShell $file $cert
    $row.ok = [bool]$row.result.ok
  } else {
    $row.result = [ordered]@{ ok = $true; skipped = "not_authenticode_signable_in_this_flow" }
    $row.ok = $true
  }
  $row.signature = Signature-Status $file
  $rows += $row
}

$attempted = @($rows | Where-Object { $_.attempted })
$failed = @($attempted | Where-Object { -not $_.ok })
$ok = $true
if ($RequireSignature -and (-not $configured)) { $ok = $false }
if ($RequireSignature -and $failed.Count -gt 0) { $ok = $false }
if ($RequireSignature -and $attempted.Count -eq 0) { $ok = $false }

$report = [ordered]@{
  ok = $ok
  policy = "project_theseus_windows_artifact_signing_v1"
  created_utc = (Get-Date).ToUniversalTime().ToString("o")
  configured = $configured
  require_signature = [bool]$RequireSignature
  signtool_path = $signTool
  certificate_path_configured = [bool]$CertificatePath
  certificate_thumbprint_configured = [bool]$CertificateThumbprint
  timestamp_url = $TimestampUrl
  path_count = $resolved.Count
  attempted_count = $attempted.Count
  failed_count = $failed.Count
  rows = $rows
  next_action = if ($ok) {
    "Windows signing completed or was intentionally skipped for non-Authenticode artifacts."
  } elseif (-not $configured) {
    "Set THESEUS_WINDOWS_SIGNING_CERT_THUMBPRINT or THESEUS_WINDOWS_SIGNING_CERT_PATH locally, then rerun with -RequireSignature."
  } elseif (-not $signTool) {
    "Install Windows SDK signtool.exe or set THESEUS_SIGNTOOL_PATH."
  } else {
    "Inspect failed rows and certificate trust chain."
  }
}

$outPath = Resolve-TheseusPath $Out
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outPath) | Out-Null
$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $outPath -Encoding UTF8
$report | ConvertTo-Json -Depth 10

if (-not $ok) { exit 2 }
