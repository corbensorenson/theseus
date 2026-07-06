param(
  [switch]$Smoke,
  [switch]$InstallBun,
  [switch]$InstallPodman,
  [string]$Out = "reports/coding_runtime_setup.json"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Test-Command {
  param([string]$Name)
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if ($cmd) {
    return @{
      available = $true
      path = $cmd.Source
    }
  }
  $localNames = @($Name, "$Name.cmd", "$Name.exe", "$Name.ps1")
  $projectTheseusTools = "D:\ProjectTheseus\tools"
  foreach ($candidateDir in @((Join-Path $projectTheseusTools $Name), (Join-Path $projectTheseusTools $Name.ToLower()), (Join-Path $projectTheseusTools $Name.ToUpper()))) {
    foreach ($localName in $localNames) {
      $localPath = Join-Path $candidateDir $localName
      if (Test-Path $localPath) {
        return @{
          available = $true
          path = $localPath
          source = "d_drive_project_toolchain"
        }
      }
    }
  }
  $toolchainRoot = Join-Path $Root "data/external_benchmark_candidates/toolchains"
  foreach ($candidateDir in @((Join-Path $toolchainRoot $Name), (Join-Path $toolchainRoot $Name.ToLower()), (Join-Path $toolchainRoot $Name.ToUpper()))) {
    foreach ($localName in $localNames) {
      $localPath = Join-Path $candidateDir $localName
      if (Test-Path $localPath) {
        return @{
          available = $true
          path = $localPath
          source = "project_local_toolchain"
        }
      }
    }
  }
  return @{
    available = $false
    path = ""
  }
}

$toolchainBunRoot = Join-Path $Root "data/external_benchmark_candidates/toolchains/bun"
$installActions = @()
if ($InstallBun -and -not (Test-Command "bun").available) {
  New-Item -ItemType Directory -Force -Path $toolchainBunRoot | Out-Null
  $proc = Start-Process -FilePath "npm" -ArgumentList @("install", "bun@1.3.14", "--prefix", $toolchainBunRoot) -NoNewWindow -PassThru -Wait
  $installActions += [ordered]@{
    tool = "bun"
    method = "npm_project_local"
    exit_code = $proc.ExitCode
    path = $toolchainBunRoot
  }
}

if ($InstallPodman -and -not (Test-Command "podman").available) {
  $winget = Get-Command winget -ErrorAction SilentlyContinue
  if ($winget) {
    $podmanInstallRoot = "D:\ProjectTheseus\tools\podman"
    New-Item -ItemType Directory -Force -Path $podmanInstallRoot | Out-Null
    $proc = Start-Process -FilePath "winget" -ArgumentList @(
      "install",
      "--id", "RedHat.Podman",
      "--silent",
      "--accept-source-agreements",
      "--accept-package-agreements",
      "--disable-interactivity",
      "--location", $podmanInstallRoot
    ) -NoNewWindow -PassThru -Wait
    $installActions += [ordered]@{
      tool = "podman"
      method = "winget_RedHat.Podman"
      exit_code = $proc.ExitCode
      note = "Container runtime installs may require a new shell, VM initialization, or a reboot before podman is visible."
    }
  } else {
    $installActions += [ordered]@{
      tool = "podman"
      method = "winget_RedHat.Podman"
      exit_code = -1
      note = "winget unavailable"
    }
  }
}

$checks = [ordered]@{
  node = Test-Command "node"
  npm = Test-Command "npm"
  bun = Test-Command "bun"
  docker = Test-Command "docker"
  podman = Test-Command "podman"
  python = Test-Command "python"
}
$containerAvailable = [bool]($checks.docker.available -or $checks.podman.available)
$nextActions = @()
if (-not $checks.bun.available) {
  $nextActions += "Bun may be installed project-locally with -InstallBun; source and metadata pressure can run without it."
}
if (-not $containerAvailable) {
  $nextActions += "Use Docker or Podman for full OpenHands/Terminal-Bench/SWE-smith container execution; source and task-contract pressure can run without it."
}
$nextActions += "Keep provider judges disabled and use only the Theseus/local OpenAI-compatible endpoint during scoring."

$cards = @(
  "source_bigcodebench",
  "source_evalplus",
  "source_livecodebench",
  "source_swe_bench",
  "source_mini_swe_agent",
  "source_swe_agent",
  "source_opencode",
  "source_opencode_bench",
  "source_openhands",
  "source_terminal_bench",
  "source_codeclash",
  "source_swe_atlas",
  "source_swe_polybench",
  "source_swe_rex",
  "source_swe_smith",
  "source_swe_gen"
)

$smokeReports = @()
if ($Smoke) {
  foreach ($card in $cards) {
    $cardPath = Join-Path $Root "benchmarks/cards/$card.json"
    if (Test-Path $cardPath) {
      $smokeOut = "reports/coding_runtime_${card}_smoke.json"
      $args = @(
        "scripts/benchmark_adapter_smoke.py",
        "--card-id", $card,
        "--out", $smokeOut,
        "--markdown-out", "reports/coding_runtime_${card}_smoke.md"
      )
      $proc = Start-Process -FilePath "python" -ArgumentList $args -NoNewWindow -PassThru -Wait
      $smokeReports += [ordered]@{
        card = $card
        exit_code = $proc.ExitCode
        report = $smokeOut
      }
    }
  }
}

$payload = [ordered]@{
  policy = "project_theseus_coding_runtime_setup_v0"
  created_utc = (Get-Date).ToUniversalTime().ToString("o")
  checks = $checks
  install_actions = $installActions
  container_runtime_available = $containerAvailable
  smoke_requested = [bool]$Smoke
  smoke_reports = $smokeReports
  safety = [ordered]@{
    external_inference_calls = 0
    provider_api_keys_required = $false
    network_during_scoring = "forbidden"
    generated_code_execution = "sandbox_required"
  }
  next_actions = $nextActions
}

$outPath = Join-Path $Root $Out
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outPath) | Out-Null
$payload | ConvertTo-Json -Depth 8 | Set-Content -Path $outPath -Encoding UTF8
$payload | ConvertTo-Json -Depth 8
