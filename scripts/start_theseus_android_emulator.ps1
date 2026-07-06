param(
  [string]$SdkRoot = "D:\ProjectTheseus\android-sdk",
  [string]$AvdHome = "D:\ProjectTheseus\android-avd",
  [string]$AvdName = "TheseusHiveApi35",
  [switch]$ColdBoot,
  [switch]$SkipAccelCheck
)

$ErrorActionPreference = "Stop"
$emulator = Join-Path $SdkRoot "emulator\emulator.exe"
if (-not (Test-Path -LiteralPath $emulator)) {
  throw "Android emulator was not found at $emulator. Run scripts\setup_theseus_android.ps1 first."
}

$env:ANDROID_HOME = $SdkRoot
$env:ANDROID_SDK_ROOT = $SdkRoot
$env:ANDROID_AVD_HOME = $AvdHome

if (-not $SkipAccelCheck) {
  $accelOutput = (& $emulator -accel-check 2>&1 | Out-String).Trim()
  $accelExit = $LASTEXITCODE
  if ($accelExit -ne 0) {
    [ordered]@{
      ok = $false
      error = "android_emulator_acceleration_unavailable"
      avd_name = $AvdName
      accel_check = $accelOutput
      next_action = "Enable CPU virtualization and install Android Emulator Hypervisor Driver or Windows Hypervisor Platform, then rerun this script."
      note = "The Android SDK and AVD can be installed on D:, but x86_64 emulator boot still requires Windows hardware acceleration."
    } | ConvertTo-Json -Depth 5
    exit 2
  }
}

$args = @("-avd", $AvdName, "-netdelay", "none", "-netspeed", "full")
if ($ColdBoot) {
  $args += "-no-snapshot-load"
}
Start-Process -FilePath $emulator -ArgumentList $args -WindowStyle Hidden

[ordered]@{
  ok = $true
  avd_name = $AvdName
  note = "Use http://10.0.2.2:8791 inside the emulator to reach the Windows host Hive node."
} | ConvertTo-Json -Depth 5
