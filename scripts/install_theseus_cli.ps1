param(
    [string]$TargetDir = "$HOME\bin",
    [switch]$SkipProbe
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if ([System.IO.Path]::IsPathRooted($TargetDir)) {
    $Target = [System.IO.Path]::GetFullPath($TargetDir)
} else {
    $Target = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $TargetDir))
}
New-Item -ItemType Directory -Force -Path $Target | Out-Null

$Wrapper = Join-Path $Target "theseus.cmd"
$Python = "py"
if (Test-Path (Join-Path $Root ".venv-puffer\Scripts\python.exe")) {
    $Python = Join-Path $Root ".venv-puffer\Scripts\python.exe"
}

@"
@echo off
cd /d "$Root"
"$Python" "$Root\scripts\theseus_cli.py" %*
"@ | Set-Content -Encoding UTF8 -Path $Wrapper

$InstallArgs = @("$Root\scripts\theseus_cli.py", "install", "--target-dir", "$Target")
if ($SkipProbe) {
    $InstallArgs += "--skip-probe"
}
& $Python @InstallArgs

Write-Host ""
Write-Host "Theseus CLI installed at $Wrapper"
Write-Host "Try: theseus status"
Write-Host "If that command is not found, add $Target to your PATH."
