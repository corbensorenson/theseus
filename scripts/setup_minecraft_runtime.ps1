param(
    [ValidateSet("bridge", "crafter", "craftax", "full-clients", "full-metadata")]
    [string]$Lane = "bridge",
    [switch]$InstallCraftax,
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONUTF8 = "1"

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Get-SourcePath {
    param([string]$SourceId)
    $Pantry = Join-Path $Root "reports\resource_pantry.json"
    if (-not (Test-Path $Pantry)) { return "" }
    $Report = Get-Content $Pantry -Raw | ConvertFrom-Json
    foreach ($Item in $Report.sources) {
        if ($Item.id -ne $SourceId) { continue }
        foreach ($Field in @("path", "clone_path", "resource_pantry_path", "staged_path")) {
            if ($Item.PSObject.Properties.Name -contains $Field) {
                $Candidate = [string]$Item.$Field
                if ($Candidate -and (Test-Path $Candidate)) {
                    return $Candidate
                }
            }
        }
    }
    return ""
}

function Ensure-Venv {
    param([string]$PythonVersion, [string]$VenvName)
    $VenvPath = Join-Path $Root $VenvName
    $PythonExe = Join-Path $VenvPath "Scripts\python.exe"
    try {
        & py "-$PythonVersion" "-c" "import sys" | Out-Null
    } catch {
        Write-Error "Python $PythonVersion is not available through the Windows py launcher."
        exit 2
    }
    if (-not (Test-Path $PythonExe)) {
        Invoke-Native py "-$PythonVersion" "-m" "venv" $VenvPath | Out-Null
    }
    Invoke-Native $PythonExe "-m" "pip" "install" "--upgrade" "pip" "setuptools<81" "wheel" | Out-Null
    return $PythonExe
}

function Install-CrafterBridge {
    param([string]$PythonExe)
    $Source = Get-SourcePath "crafter"
    if (-not $Source) { Write-Error "Crafter source is not staged in reports/resource_pantry.json."; exit 3 }
    Invoke-Native $PythonExe "-m" "pip" "install" "-e" $Source
}

function Install-CraftaxBridge {
    param([string]$PythonExe)
    $Source = Get-SourcePath "craftax"
    if (-not $Source) { Write-Error "Craftax source is not staged in reports/resource_pantry.json."; exit 3 }
    Invoke-Native $PythonExe "-m" "pip" "install" "-e" $Source
}

function Install-MalmoClient {
    param([string]$PythonExe)
    $Source = Get-SourcePath "malmo"
    if (-not $Source) { Write-Error "Malmo source is not staged in reports/resource_pantry.json."; exit 3 }
    $MalmoEnv = Join-Path $Source "MalmoEnv"
    if (-not (Test-Path $MalmoEnv)) { Write-Error "MalmoEnv package was not found under $Source."; exit 3 }
    Invoke-Native $PythonExe "-m" "pip" "install" "-e" $MalmoEnv
}

function Install-MineDojoClient {
    param([string]$PythonExe)
    $Source = Get-SourcePath "minedojo"
    if (-not $Source) { Write-Error "MineDojo source is not staged in reports/resource_pantry.json."; exit 3 }
    Invoke-Native $PythonExe "-m" "pip" "install" `
        "gym==0.23.1" "pyyaml" "jinja2" "lxml" "coloredlogs" "xmltodict" "Pyro4" `
        "psutil" "opencv-python" "multiprocess" "pytest" "daemoniker" "tqdm" `
        "requests" "mypy-extensions" "jsonlines" "praw" "wget" `
        "importlib_resources" "hydra-core" "Pillow"
    Invoke-Native $PythonExe "-m" "pip" "install" "--no-build-isolation" "--no-deps" "-e" $Source
}

Write-Host "Project Theseus Minecraft/Open-World RL runtime setup"
Write-Host "Lane: $Lane"
Write-Host "This installs only open-source benchmark/runtime packages into .venv-minecraft-rl-py311."
Write-Host "It does not read launcher credentials, join public servers, or download commercial game assets."

$PythonExe = Ensure-Venv "3.11" ".venv-minecraft-rl-py311"

switch ($Lane) {
    "bridge" {
        Install-CrafterBridge $PythonExe
        if ($InstallCraftax) { Install-CraftaxBridge $PythonExe }
    }
    "crafter" {
        Install-CrafterBridge $PythonExe
    }
    "craftax" {
        Install-CraftaxBridge $PythonExe
    }
    "full-clients" {
        Install-CrafterBridge $PythonExe
        Install-CraftaxBridge $PythonExe
        Install-MineDojoClient $PythonExe
        Install-MalmoClient $PythonExe
    }
    "full-metadata" {
        Install-CrafterBridge $PythonExe
        Install-CraftaxBridge $PythonExe
        Install-MineDojoClient $PythonExe
        Install-MalmoClient $PythonExe
        Write-Host "Installed full local Minecraft client imports. Game execution remains local-license and loopback-policy gated."
    }
}

Invoke-Native $PythonExe "scripts\minecraft_runtime_probe.py" "--out" "reports\minecraft_runtime_probe.json"

if (-not $SkipSmoke) {
    Invoke-Native $PythonExe `
        "scripts\benchmark_adapter_smoke.py" `
        "--card-id" "source_crafter" `
        "--card-id" "source_craftax" `
        "--card-id" "source_minedojo" `
        "--card-id" "source_malmo" `
        "--card-id" "source_voyager_minecraft" `
        "--card-id" "source_minerl" `
        "--out" "reports\benchmark_adapter_smoke_status.json" `
        "--markdown-out" "reports\benchmark_adapter_smoke_status.md"
}

Write-Host "Minecraft/Open-World RL runtime setup finished."
