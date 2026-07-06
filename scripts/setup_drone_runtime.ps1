param(
    [ValidateSet("competition", "pyflyt", "gym-pybullet", "racing-client", "control-client", "all-dev")]
    [string]$Lane = "competition",
    [switch]$UsePython311DevFallback,
    [switch]$InstallOptionalClients,
    [switch]$InstallTrainingExtras,
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

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
    Invoke-Native $PythonExe "-m" "pip" "install" "--upgrade" "pip" "setuptools" "wheel" | Out-Null
    return $PythonExe
}

function Install-Lane {
    param([string]$LaneName)
    switch ($LaneName) {
        "competition" {
            $PythonVersion = if ($UsePython311DevFallback) { "3.11" } else { "3.14" }
            $VenvName = if ($UsePython311DevFallback) { ".venv-drone-py311-dev" } else { ".venv-drone-py314" }
            $PythonExe = Ensure-Venv $PythonVersion $VenvName
            Invoke-Native $PythonExe "-m" "pip" "install" "numpy" "opencv-python" "mavsdk"
        }
        "pyflyt" {
            $PythonExe = Ensure-Venv "3.11" ".venv-drone-pyflyt-py311"
            $Source = Get-SourcePath "pyflyt"
            if (-not $Source) { Write-Error "PyFlyt source is not staged in reports/resource_pantry.json."; exit 3 }
            Invoke-Native $PythonExe "-m" "pip" "install" "-e" $Source
        }
        "gym-pybullet" {
            $PythonExe = Ensure-Venv "3.11" ".venv-drone-gym-pybullet-py311"
            $Source = Get-SourcePath "gym_pybullet_drones"
            if (-not $Source) { Write-Error "gym-pybullet-drones source is not staged in reports/resource_pantry.json."; exit 3 }
            Invoke-Native $PythonExe "-m" "pip" "install" "setuptools<81" "numpy>=2.2" "scipy>=1.15" "transforms3d>=0.4" "pybullet>=3.2.7" "gymnasium>=1.2" "control>=0.10.2" "matplotlib"
            Invoke-Native $PythonExe "-m" "pip" "install" "--no-deps" "-e" $Source
            if ($InstallTrainingExtras) {
                Invoke-Native $PythonExe "-m" "pip" "install" "stable-baselines3"
            }
        }
        "racing-client" {
            $PythonExe = Ensure-Venv "3.11" ".venv-drone-racing-py311"
            Invoke-Native $PythonExe "-m" "pip" "install" "airsimdroneracinglab" "opencv-python"
            Invoke-Native $PythonExe "-m" "pip" "install" "--no-build-isolation" "airsim"
        }
        "control-client" {
            $PythonExe = Ensure-Venv "3.11" ".venv-drone-control-py311"
            Invoke-Native $PythonExe "-m" "pip" "install" "mavsdk"
            if ($InstallOptionalClients) {
                Invoke-Native $PythonExe "-m" "pip" "install" "numpy" "msgpack-rpc-python" "msgpack-python" "tornado"
                Invoke-Native $PythonExe "-m" "pip" "install" "--no-build-isolation" "airsim"
            }
        }
    }
}

Write-Host "Project Theseus drone runtime setup"
Write-Host "Lane: $Lane"
if ($Lane -eq "all-dev") {
    Install-Lane "pyflyt"
    Install-Lane "gym-pybullet"
    Install-Lane "control-client"
    if ($InstallOptionalClients) {
        Install-Lane "racing-client"
    }
} else {
    Install-Lane $Lane
}

Invoke-Native "python" "scripts\python_runtime_compatibility.py" "--out" "reports\python_runtime_compatibility.json"

if (-not $SkipSmoke) {
    Invoke-Native "python" `
        "scripts\benchmark_adapter_smoke.py" `
        "--card-id" "source_pyflyt" `
        "--card-id" "source_gym_pybullet_drones" `
        "--card-id" "source_mavsdk_python" `
        "--out" "reports\benchmark_adapter_smoke_status.json" `
        "--markdown-out" "reports\benchmark_adapter_smoke_status.md"
}

Write-Host "Drone runtime setup finished. For official AI Grand Prix runs, reports/python_runtime_compatibility.json must show ai_grand_prix_runtime_ready=true."
