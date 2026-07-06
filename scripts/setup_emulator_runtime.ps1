param(
    [ValidateSet("gb-gbc", "gba", "stable-retro", "all-dev")]
    [string]$Lane = "all-dev",
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

function Invoke-BestEffort {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Body
    )
    try {
        & $Body
        Write-Host "${Name}: ok"
        return $true
    } catch {
        Write-Warning "${Name}: $($_.Exception.Message)"
        return $false
    }
}

function Get-SourcePath {
    param([string]$SourceId)
    $Pantry = Join-Path $Root "reports\resource_pantry.json"
    if (Test-Path $Pantry) {
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
    }
    $Fallback = Join-Path "D:\ProjectTheseus\resource_pantry\git" $SourceId
    if (Test-Path $Fallback) {
        return $Fallback
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

function Install-GbGbcRuntime {
    param([string]$PythonExe)
    Invoke-Native $PythonExe "-m" "pip" "install" "gymboy"
}

function Install-GbaRuntime {
    param([string]$PythonExe)
    $PyGbaSource = Get-SourcePath "pygba"
    if (-not $PyGbaSource) { Write-Error "PyGBA source is not staged in reports/resource_pantry.json."; exit 3 }
    Invoke-Native $PythonExe "-m" "pip" "install" "cffi" "cached-property" "gymnasium" "numpy" "pygame" "ziglang"
    Invoke-Native $PythonExe "-m" "pip" "install" "-e" $PyGbaSource

    $MgbaSource = Get-SourcePath "mgba"
    if (-not $MgbaSource) {
        Write-Warning "mGBA source is not staged; PyGBA import will remain runtime-blocked."
        return
    }
    Invoke-Native $PythonExe "scripts\patch_mgba_windows_builder.py" "--source" $MgbaSource
    $VsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $VsDevCmd)) {
        Write-Warning "Visual Studio developer shell not found; mGBA Python CFFI rebuild skipped."
        return
    }
    $BuildDir = Join-Path $MgbaSource "build-python311"
    $PythonPackage = Join-Path $MgbaSource "src\platform\python"
    $BuildInclude = (Join-Path $BuildDir "include").Replace("\", "/")
    $BuildRelease = (Join-Path $BuildDir "Release").Replace("\", "/")
    $BuildPyRelease = (Join-Path $BuildDir "python\Release").Replace("\", "/")
    Invoke-BestEffort "mGBA Python CFFI rebuild" {
        $Cmd = @(
            "`"$VsDevCmd`" -arch=x64 >nul",
            "set `"CPP=cl.exe /EP`"",
            "set `"BINDIR=$BuildRelease`"",
            "set `"LIBDIR=$BuildRelease`"",
            "set `"PYLIBDIR=$BuildPyRelease`"",
            "set `"CPPFLAGS=/I$BuildInclude`"",
            "`"$PythonExe`" -m pip install --no-build-isolation -e `"$PythonPackage`""
        ) -join " && "
        & cmd.exe /s /c $Cmd
        if ($LASTEXITCODE -ne 0) {
            throw "mGBA CFFI build exited with $LASTEXITCODE"
        }
    } | Out-Null
}

function Install-StableRetroRuntime {
    param([string]$PythonExe)
    Invoke-BestEffort "stable-retro Windows build" {
        Invoke-Native $PythonExe "-m" "pip" "install" "stable-retro"
    } | Out-Null
}

Write-Host "Project Theseus emulator runtime setup"
Write-Host "Lane: $Lane"
Write-Host "ROM bytes remain user-supplied local assets; this script never downloads ROMs."

$PythonExe = Ensure-Venv "3.11" ".venv-emulator-py311"
if ($Lane -in @("gb-gbc", "all-dev")) { Install-GbGbcRuntime $PythonExe }
if ($Lane -in @("gba", "all-dev")) { Install-GbaRuntime $PythonExe }
if ($Lane -in @("stable-retro", "all-dev")) { Install-StableRetroRuntime $PythonExe }

if (-not $SkipSmoke) {
    Invoke-Native "python" `
        "scripts\benchmark_adapter_smoke.py" `
        "--card-id" "source_gymboy" `
        "--card-id" "source_pyboy" `
        "--card-id" "source_stable_retro" `
        "--card-id" "source_pygba" `
        "--card-id" "local_rom_gba_pokemon_emerald" `
        "--out" "reports\benchmark_adapter_smoke_status.json" `
        "--markdown-out" "reports\benchmark_adapter_smoke_status.md"
}

Write-Host "Emulator runtime setup finished. Any remaining GBA block is now an mGBA/PyGBA native runtime block, not a missing setup script."
