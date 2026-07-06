param(
  [string]$SdkRoot = "D:\ProjectTheseus\android-sdk",
  [string]$ToolsRoot = "D:\ProjectTheseus\android-tools",
  [string]$JdkRoot = "D:\ProjectTheseus\android-tools\jdk-17",
  [string]$AvdHome = "D:\ProjectTheseus\android-avd",
  [string]$ApiLevel = "35",
  [string]$BuildToolsVersion = "35.0.0",
  [string]$GradleVersion = "8.7",
  [string]$CommandLineToolsRevision = "14742923",
  [string]$AvdName = "TheseusHiveApi$ApiLevel",
  [switch]$SkipEmulator,
  [switch]$AcceptLicenses,
  [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

function Ensure-Directory([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
  }
}

function Download-File([string]$Url, [string]$OutFile) {
  if (Test-Path -LiteralPath $OutFile) { return }
  Write-Host "Downloading $Url"
  Invoke-WebRequest -Uri $Url -OutFile $OutFile
}

function Expand-Zip([string]$ZipPath, [string]$Destination) {
  if (Test-Path -LiteralPath $Destination) { return }
  Ensure-Directory (Split-Path -Parent $Destination)
  $tmp = "$Destination.tmp"
  if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force }
  Expand-Archive -LiteralPath $ZipPath -DestinationPath $tmp -Force
  $children = Get-ChildItem -LiteralPath $tmp
  if ($children.Count -eq 1 -and $children[0].PSIsContainer) {
    Move-Item -LiteralPath $children[0].FullName -Destination $Destination
    Remove-Item -LiteralPath $tmp -Recurse -Force
  } else {
    Move-Item -LiteralPath $tmp -Destination $Destination
  }
}

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$androidProject = Join-Path $repo "android\TheseusHive"
$downloads = Join-Path $ToolsRoot "downloads"
$jdkHome = $JdkRoot
$gradleHome = Join-Path $ToolsRoot "gradle-$GradleVersion"
$cmdlineRoot = Join-Path $SdkRoot "cmdline-tools"
$cmdlineLatest = Join-Path $cmdlineRoot "latest"

Ensure-Directory $SdkRoot
Ensure-Directory $ToolsRoot
Ensure-Directory $AvdHome
Ensure-Directory $downloads
Ensure-Directory $cmdlineRoot

$jdkZip = Join-Path $downloads "temurin-jdk17-windows-x64.zip"
Download-File "https://api.adoptium.net/v3/binary/latest/17/ga/windows/x64/jdk/hotspot/normal/eclipse?project=jdk" $jdkZip
Expand-Zip $jdkZip $jdkHome

$gradleZip = Join-Path $downloads "gradle-$GradleVersion-bin.zip"
Download-File "https://services.gradle.org/distributions/gradle-$GradleVersion-bin.zip" $gradleZip
Expand-Zip $gradleZip $gradleHome

$cmdlineZip = Join-Path $downloads "commandlinetools-win-$CommandLineToolsRevision.zip"
$cmdlineMarker = Join-Path $cmdlineLatest ".theseus_revision"
$installedCmdlineRevision = if (Test-Path -LiteralPath $cmdlineMarker) { (Get-Content -LiteralPath $cmdlineMarker -Raw).Trim() } else { "" }
Download-File "https://dl.google.com/android/repository/commandlinetools-win-$CommandLineToolsRevision`_latest.zip" $cmdlineZip
if ((-not (Test-Path -LiteralPath $cmdlineLatest)) -or ($installedCmdlineRevision -ne $CommandLineToolsRevision)) {
  if (Test-Path -LiteralPath $cmdlineLatest) {
    Remove-Item -LiteralPath $cmdlineLatest -Recurse -Force
  }
  $tmp = Join-Path $downloads "cmdline-tools-expanded"
  if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force }
  Expand-Archive -LiteralPath $cmdlineZip -DestinationPath $tmp -Force
  $inner = Join-Path $tmp "cmdline-tools"
  Move-Item -LiteralPath $inner -Destination $cmdlineLatest
  Remove-Item -LiteralPath $tmp -Recurse -Force
  $CommandLineToolsRevision | Set-Content -Path $cmdlineMarker -Encoding ASCII
}

$env:ANDROID_HOME = $SdkRoot
$env:ANDROID_SDK_ROOT = $SdkRoot
$env:ANDROID_AVD_HOME = $AvdHome
$env:JAVA_HOME = $jdkHome
$env:PATH = "$jdkHome\bin;$cmdlineLatest\bin;$SdkRoot\platform-tools;$SdkRoot\emulator;$gradleHome\bin;$env:PATH"

$sdkmanager = Join-Path $cmdlineLatest "bin\sdkmanager.bat"
$avdmanager = Join-Path $cmdlineLatest "bin\avdmanager.bat"
$gradle = Join-Path $gradleHome "bin\gradle.bat"

if ($AcceptLicenses) {
  "y`ny`ny`ny`ny`ny`ny`ny`ny`ny" | & $sdkmanager --licenses | Out-Host
}

$packages = @(
  "platform-tools",
  "platforms;android-$ApiLevel",
  "build-tools;$BuildToolsVersion"
)
if (-not $SkipEmulator) {
  $packages += @(
    "emulator",
    "system-images;android-$ApiLevel;google_apis;x86_64"
  )
}
& $sdkmanager @packages

$localProperties = Join-Path $androidProject "local.properties"
"sdk.dir=$($SdkRoot -replace '\\','/')" | Set-Content -Path $localProperties -Encoding ASCII

if (-not $SkipEmulator) {
  $existingAvds = (& $avdmanager list avd) -join "`n"
  if ($existingAvds -notmatch [regex]::Escape("Name: $AvdName")) {
    "no" | & $avdmanager create avd -n $AvdName -k "system-images;android-$ApiLevel;google_apis;x86_64" --device "pixel_6"
  }
}

$emulatorAccelerationOk = $null
$emulatorAccelCheck = ""
if (-not $SkipEmulator) {
  $emulator = Join-Path $SdkRoot "emulator\emulator.exe"
  if (Test-Path -LiteralPath $emulator) {
    $emulatorAccelCheck = (& $emulator -accel-check 2>&1 | Out-String).Trim()
    $emulatorAccelerationOk = ($LASTEXITCODE -eq 0)
  }
}

if (-not $NoBuild) {
  & $gradle -p $androidProject assembleDebug
}

$report = [ordered]@{
  ok = $true
  sdk_root = $SdkRoot
  tools_root = $ToolsRoot
  jdk_root = $jdkHome
  commandline_tools_revision = $CommandLineToolsRevision
  avd_home = $AvdHome
  avd_name = if ($SkipEmulator) { "" } else { $AvdName }
  emulator_acceleration_ok = $emulatorAccelerationOk
  emulator_accel_check = $emulatorAccelCheck
  emulator_next_action = if ($emulatorAccelerationOk -eq $false) { "Enable CPU virtualization and install Android Emulator Hypervisor Driver or Windows Hypervisor Platform before booting the AVD." } else { "" }
  android_project = $androidProject
  debug_apk = Join-Path $androidProject "app\build\outputs\apk\debug\app-debug.apk"
}
$report | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $repo "reports\android_hive_setup.json") -Encoding UTF8
$report | ConvertTo-Json -Depth 5
