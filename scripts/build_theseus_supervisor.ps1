param(
  [switch]$Release
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$args = @("build", "-p", "theseus-supervisor")
if ($Release) {
  $args += "--release"
}

cargo @args

Write-Host "Built Project Theseus supervisor."
Write-Host "Run: cargo run -p theseus-supervisor -- doctor"
