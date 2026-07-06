param(
  [string]$SdkRoot = "D:\ProjectTheseus\android-sdk",
  [string]$ToolsRoot = "D:\ProjectTheseus\android-tools",
  [string]$JdkRoot = "D:\ProjectTheseus\android-tools\jdk-17",
  [string]$GradleVersion = "8.7",
  [switch]$InstallDebug,
  [string]$DeviceSerial = ""
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$androidProject = Join-Path $repo "android\TheseusHive"
$gradle = Join-Path $ToolsRoot "gradle-$GradleVersion\bin\gradle.bat"

if (-not (Test-Path -LiteralPath $gradle)) {
  throw "Gradle was not found at $gradle. Run scripts\setup_theseus_android.ps1 first."
}
if (-not (Test-Path -LiteralPath $SdkRoot)) {
  throw "Android SDK was not found at $SdkRoot. Run scripts\setup_theseus_android.ps1 first."
}
if (-not (Test-Path -LiteralPath (Join-Path $JdkRoot "bin\java.exe"))) {
  throw "JDK 17 was not found at $JdkRoot. Run scripts\setup_theseus_android.ps1 first."
}

$env:ANDROID_HOME = $SdkRoot
$env:ANDROID_SDK_ROOT = $SdkRoot
$env:JAVA_HOME = $JdkRoot
$env:PATH = "$JdkRoot\bin;$SdkRoot\platform-tools;$SdkRoot\emulator;$env:PATH"

$localProperties = Join-Path $androidProject "local.properties"
"sdk.dir=$($SdkRoot -replace '\\','/')" | Set-Content -Path $localProperties -Encoding ASCII

& $gradle -p $androidProject assembleDebug

$apk = Join-Path $androidProject "app\build\outputs\apk\debug\app-debug.apk"
if ($InstallDebug) {
  $adb = Join-Path $SdkRoot "platform-tools\adb.exe"
  $serialArgs = @()
  if ($DeviceSerial) {
    $serialArgs = @("-s", $DeviceSerial)
  }
  & $adb @serialArgs install -r $apk
}

[ordered]@{
  ok = $true
  apk = $apk
  installed = [bool]$InstallDebug
} | ConvertTo-Json -Depth 5
